from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import os.path as osp
import cv2
import logging
import argparse
import motmetrics as mm
import numpy as np
import torch

from tracker.multitracker import JDETracker
from tracking_utils import visualization as vis
from tracking_utils.log import logger
from tracking_utils.timer import Timer
from tracking_utils.evaluation import Evaluator
import datasets.dataset.jde as datasets
from frog.optimize_location import optimize
from frog.calculate_roi import FrogROI
from frog.calculate_roi import blur_image, visualize_rois, visualize_all_regions
from frog.foveation import foveation_tlwh, jde_letterbox
from frog.foveation import foveation_snake
from tracking_utils.utils import mkdir_if_missing
from opts import opts


FOVEA_OPTIMIZE = False


def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def write_results_score(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},{s},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h, s=score)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def eval_seq(opt, dataloader, data_type, result_filename,
              save_dir=None, show_image=True, frame_rate=30, use_cuda=True,
              sequence_name='default', clear_prev_vis=False):
    if save_dir:
        mkdir_if_missing(save_dir)
    tracker = JDETracker(opt, frame_rate=frame_rate)
    timer = Timer()
    results = []
    frame_id = 0
    prev_online_tlwhs = []
    fovea_visualize_path = opt.fovea_visualize_path
    fovea_factor = opt.fovea_factor

    # clear all .jpg images in fovea_visualize_path
    if opt.visualize_fovea > 0 and clear_prev_vis:
        for file in os.listdir(fovea_visualize_path):
            if file.endswith('.jpg'):
                os.remove(os.path.join(fovea_visualize_path, file))

    #for path, img, img0 in dataloader:
    for i, (path, img, img0) in enumerate(dataloader):

        #if i % 8 != 0:
            #continue
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        if opt.fovea_optimize:
            # Frog undergrad thesis algorithm settings
            img_height, img_width = img0.shape[:2]

            if i < 1:
                region_detector = FrogROI(image_width=img_width, image_height=img_height,
                                          init_image_path=path, region_scale=0.025, pixel_change_threshold=150)
                blurred_img = blur_image(img0, 37)
                region_detector.store_engram_images_raw(blurred_img)
            


            fovea_width = int(img_width // fovea_factor)
            fovea_height = int(img_height // fovea_factor)
            init_x, init_y = img_width // 2, img_height // 2
            epochs = 6
            algo = 'annealing'
            visualize = False
            visualize_path = '../results'

            # additional roi based on adaptation
            if i >= 1:
                # roi based on adaptation is only valid after the first frame
                blurred_img = blur_image(img0, 37)
                roi_tlwhs = region_detector.compare_engram(blurred_img.astype(np.float32))
                if len(roi_tlwhs) > 0:
                    # add all items in roi_tlwhs to prev_online_tlwhs
                    for roi_tlwh in roi_tlwhs:
                        prev_online_tlwhs.append(roi_tlwh)
                # add this frame to engram image list
                region_detector.store_engram_images_raw(blurred_img)
                # calculate engram for next frame
                region_detector.calculate_engram()


            if len(prev_online_tlwhs) > 0:
                fovea_x, fovea_y = optimize(prev_online_tlwhs, fovea_width, fovea_height, img_width, img_height,
                                   init_x=init_x, init_y=init_y, epochs=epochs, algo=algo,
                                   visualize=visualize, visualize_path=visualize_path)
                if opt.visualize_fovea > 0 and i % 50 == 0:
                    print(f'Optimized position: x={fovea_x:.2f}, y={fovea_y:.2f}')
                    # make a copy of blurred_img and region_detector.engram
                    if opt.visualize_fovea > 1:
                        blurred_img_copy = np.copy(blurred_img)
                        origin_img_copy = np.copy(img0)
                        engram_copy = np.copy(region_detector.engram)
                        print(f'Saving images with marked roi...')
                        visualize_rois(blurred_img_copy, origin_img_copy, engram_copy, roi_tlwhs, 
                                    result_path='../fovea_result/', file_marker=str(i))
                    
                fovea_img0 = foveation_tlwh(img0, [int(fovea_x), int(fovea_y), fovea_width, fovea_height], blur_factor=37)
            else:
                print(f'Empty tlwhs list at frame {frame_id}, using default fovea location')
                # 否则默认最中央区域清晰，剩余区域模糊
                init_top = init_x - fovea_width // 2
                init_left = init_y - fovea_height // 2
                fovea_img0 = foveation_tlwh(img0, [int(init_top), int(init_left), fovea_width, fovea_height], blur_factor=37)
            
            if opt.visualize_fovea > 1 and i % 5 == 0:
                # save origin blurred image for drawing all target regions
                img0_blurred = blur_image(img0, 37)

            img0 = fovea_img0
            # 使用原作者代码中的宽高设置
            img, _, _, _ = jde_letterbox(img0, height=opt.img_size[1], width=opt.img_size[0])
            # Normalize RGB
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img, dtype=np.float32)
            img /= 255.0

            if opt.visualize_fovea > 0 and i % 50 == 0:
                fovea_path = opt.fovea_visualize_path
                cv2.imwrite(fovea_path + f'/{sequence_name}_{frame_id}_opt_foveated.jpg', fovea_img0)
                print(f'Saving foveated image at frame {frame_id} to {fovea_path}')
        
            # 清空prev_online_tlwhs
            prev_online_tlwhs = []

        if opt.static_fovea:
            img_height, img_width = img0.shape[:2]
            fovea_width = int(img_width // fovea_factor)
            fovea_height = int(img_height // fovea_factor)
            fovea_x = img_width // 2 - fovea_width // 2
            fovea_y = img_height // 2 - fovea_height // 2
            
            fovea_img0 = foveation_tlwh(img0, [int(fovea_x), int(fovea_y), fovea_width, fovea_height], blur_factor=37)

            img0 = fovea_img0
            # 使用原作者代码中的宽高设置
            img, _, _, _ = jde_letterbox(img0, height=opt.img_size[1], width=opt.img_size[0])
            # Normalize RGB
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img, dtype=np.float32)
            img /= 255.0

            if opt.visualize_fovea and i % 50 == 0:
                fovea_path = opt.fovea_visualize_path
                cv2.imwrite(fovea_path + f'/{sequence_name}_{frame_id}_static_foveated.jpg', fovea_img0)
                print(f'Saving static foveated image at frame {frame_id} to {fovea_path}')

        if opt.snake_fovea != 0:
            img_height, img_width = img0.shape[:2]
            fovea_width = int(img_width // fovea_factor)
            fovea_height = int(img_height // fovea_factor)

            fovea_img0 = foveation_snake(img0, i, fovea_width, fovea_height, blur_factor=37, mode=opt.snake_fovea)

            img0 = fovea_img0
            # 使用原作者代码中的宽高设置
            img, _, _, _ = jde_letterbox(img0, height=opt.img_size[1], width=opt.img_size[0])
            # Normalize RGB
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img, dtype=np.float32)
            img /= 255.0

            if opt.visualize_fovea and i % 50 == 0:
                fovea_path = opt.fovea_visualize_path
                cv2.imwrite(fovea_path + f'/{sequence_name}_{frame_id}_snake_foveated_1.jpg', fovea_img0)
                print(f'Saving snake foveated image(mode {opt.snake_fovea}) at frame {frame_id} to {fovea_path}')
        
        # run tracking
        timer.tic()
        if use_cuda:
            blob = torch.from_numpy(img).cuda().unsqueeze(0)
        else:
            blob = torch.from_numpy(img).unsqueeze(0)
        online_targets = tracker.update(blob, img0)
        online_tlwhs = []
        online_ids = []
        #online_scores = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                if opt.fovea_optimize:
                    # 将本帧中得到的所有目标的锚框信息（tlwhs）加入到prev_online_tlwhs中，以便下一帧计算中央凹位置
                    prev_online_tlwhs.append(tlwh)
                #online_scores.append(t.score)
        
        if opt.fovea_optimize:
            prev_online_tlwhs = online_tlwhs
            
            if i % 5 == 0 and i > 0 and opt.visualize_fovea > 1:
                visualize_all_regions(img0_blurred, roi_tlwhs, online_tlwhs,
                                      result_path='../fovea_result/', file_marker=str(i))
        
        # if frame_id % 20 == 0:
        #     print(f'Example of tlwhs values: {online_tlwhs[0][:5]}')
        #     print(f'FOVEA_OPTIMIZE value is {FOVEA_OPTIMIZE}')
        
        
            
        timer.toc()
        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))
        #results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
        if show_image or save_dir is not None:
            online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                          fps=1. / timer.average_time)
        if show_image:
            cv2.imshow('online_im', online_im)
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
        frame_id += 1
    # save results
    write_results(result_filename, results, data_type)
    #write_results_score(result_filename, results, data_type)
    return frame_id, timer.average_time, timer.calls


def main(opt, data_root='/data/MOT16/train', det_root=None, seqs=('MOT16-05',), exp_name='demo',
         save_images=False, save_videos=False, show_image=True):
    logger.setLevel(logging.INFO)
    result_root = os.path.join(data_root, '..', 'results', exp_name)
    mkdir_if_missing(result_root)
    data_type = 'mot'

    # Frog undergrad thesis algorithm settings
    global FOVEA_OPTIMIZE
    FOVEA_OPTIMIZE= opt.fovea_optimize
    if FOVEA_OPTIMIZE:
        logger.info(f'[Frog] Fovea optimization is enabled')
    
    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    for seq in seqs:
        output_dir = os.path.join(data_root, '..', 'outputs', exp_name, seq) if save_images or save_videos else None
        logger.info('start seq: {}'.format(seq))
        dataloader = datasets.LoadImages(osp.join(data_root, seq, 'img1'), opt.img_size)
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        meta_info = open(os.path.join(data_root, seq, 'seqinfo.ini')).read()
        frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
        nf, ta, tc = eval_seq(opt, dataloader, data_type, result_filename,
                              save_dir=output_dir, show_image=show_image, frame_rate=frame_rate, sequence_name=seq)
        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)

        # eval
        logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(data_root, seq, data_type)
        accs.append(evaluator.eval_file(result_filename))
        if save_videos:
            output_video_path = osp.join(output_dir, '{}.mp4'.format(seq))
            cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(output_dir, output_video_path)
            os.system(cmd_str)
    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    Evaluator.save_summary(summary, os.path.join(result_root, 'summary_{}.xlsx'.format(exp_name)))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    opt = opts().init()

    if not opt.val_mot16:
        # seqs_str = '''Venice-2
        #               KITTI-13
        #               KITTI-17
        #               ETH-Bahnhof
        #               ETH-Sunnyday
        #               PETS09-S2L1
        #               TUD-Campus
        #               TUD-Stadtmitte
        #               ADL-Rundle-6
        #               ADL-Rundle-8
        #               ETH-Pedcross2
        #               TUD-Stadtmitte'''
        seqs_str = '''TUD-Campus
                      PETS09-S2L1'''
        data_root = os.path.join(opt.data_dir, 'MOT15-Fair/images/train')
    else:
        seqs_str = '''MOT16-02
                      MOT16-04
                      MOT16-05
                      MOT16-09
                      MOT16-10
                      MOT16-11
                      MOT16-13'''
        data_root = os.path.join(opt.data_dir, 'MOT16/train')
    if opt.test_mot16:
        seqs_str = '''MOT16-01
                      MOT16-03
                      MOT16-06
                      MOT16-07
                      MOT16-08
                      MOT16-12
                      MOT16-14'''
        #seqs_str = '''MOT16-01 MOT16-07 MOT16-12 MOT16-14'''
        #seqs_str = '''MOT16-06 MOT16-08'''
        data_root = os.path.join(opt.data_dir, 'MOT16/test')
    if opt.test_mot15:
        seqs_str = '''ADL-Rundle-1
                      ADL-Rundle-3
                      AVG-TownCentre
                      ETH-Crossing
                      ETH-Jelmoli
                      ETH-Linthescher
                      KITTI-16
                      KITTI-19
                      PETS09-S2L2
                      TUD-Crossing
                      Venice-1'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/test')
    if opt.test_mot17:
        seqs_str = '''MOT17-01-SDP
                      MOT17-03-SDP
                      MOT17-06-SDP
                      MOT17-07-SDP
                      MOT17-08-SDP
                      MOT17-12-SDP
                      MOT17-14-SDP'''
        data_root = os.path.join(opt.data_dir, 'MOT17/images/test')
    if opt.val_mot17:
        seqs_str = '''MOT17-02-SDP
                      MOT17-04-SDP
                      MOT17-05-SDP
                      MOT17-09-SDP
                      MOT17-10-SDP
                      MOT17-11-SDP
                      MOT17-13-SDP'''
        data_root = os.path.join(opt.data_dir, 'MOT17/images/train')
    if opt.val_mot15:
        seqs_str = '''Venice-2
                      KITTI-13
                      KITTI-17
                      ETH-Bahnhof
                      ETH-Sunnyday
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte
                      ADL-Rundle-6
                      ADL-Rundle-8
                      ETH-Pedcross2
                      TUD-Stadtmitte'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/train')
    if opt.val_mot20:
        seqs_str = '''MOT20-01
                      MOT20-02
                      MOT20-03
                      MOT20-05
                      '''
        data_root = os.path.join(opt.data_dir, 'MOT20/images/train')
    if opt.test_mot20:
        seqs_str = '''MOT20-04
                      MOT20-06
                      MOT20-07
                      MOT20-08
                      '''
        data_root = os.path.join(opt.data_dir, 'MOT20/images/test')
    seqs = [seq.strip() for seq in seqs_str.split()]

    print(f'Data root is {data_root}')

    main(opt,
         data_root=data_root,
         seqs=seqs,
         exp_name='snake_fovea_sample',
         show_image=False,
         save_images=False,
         save_videos=True)


# Use the following command to run expreiments:
    # Optimized foveation
    # python track.py mot --load_model ../models/fairmot_dla34.pth --conf_thres 0.6 --fovea_optimize=True 

    # Static foveation
    # python track.py mot --load_model ../models/fairmot_dla34.pth --conf_thres 0.6 --static_fovea=True


    # Snake foveation(width first)
    # python track.py mot --load_model ../models/fairmot_dla34.pth --conf_thres 0.6 --snake_fovea=1


    
    # To add visualization of preprocessed images, set visualize_fovea to 1
    # --visualize_fovea=1

    # To add visualization of engrams, set visualize_fovea to 2
    # --visualize_fovea=2
