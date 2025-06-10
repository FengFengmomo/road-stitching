import warnings
from types import SimpleNamespace

import numpy as np

from .blender import Blender
from .camera_adjuster import CameraAdjuster
from .camera_estimator import CameraEstimator
from .camera_wave_corrector import WaveCorrector
from .cropper import Cropper
from .exposure_error_compensator import ExposureErrorCompensator
from .feature_detector import FeatureDetector
from .feature_matcher import FeatureMatcher
from .images import Images
from .seam_finder import SeamFinder
from .stitching_error import StitchingError, StitchingWarning
from .subsetter import Subsetter
from .timelapser import Timelapser
from .verbose import verbose_stitching
from .warper import Warper
from .sensor_seam_finder import SensorSeamFinder
from .warper_adjuster import WarperAdjuster

import time
import cv2
import os


class Stitcher:
    DEFAULT_SETTINGS = {
        "medium_megapix": Images.Resolution.MEDIUM.value, # 图像放缩到中等水平
        "detector": FeatureDetector.DEFAULT_DETECTOR,
        "nfeatures": 500,
        "matcher_type": FeatureMatcher.DEFAULT_MATCHER,
        "range_width": FeatureMatcher.DEFAULT_RANGE_WIDTH,
        "try_use_gpu": True,
        "match_conf": None,
        "confidence_threshold": Subsetter.DEFAULT_CONFIDENCE_THRESHOLD,
        "matches_graph_dot_file": Subsetter.DEFAULT_MATCHES_GRAPH_DOT_FILE,
        "estimator": CameraEstimator.DEFAULT_CAMERA_ESTIMATOR,
        "adjuster": CameraAdjuster.DEFAULT_CAMERA_ADJUSTER,
        "refinement_mask": CameraAdjuster.DEFAULT_REFINEMENT_MASK,
        "wave_correct_kind": WaveCorrector.DEFAULT_WAVE_CORRECTION,
        "warper_type": Warper.DEFAULT_WARP_TYPE,
        "low_megapix": Images.Resolution.LOW.value,
        "crop": Cropper.DEFAULT_CROP,
        "compensator": ExposureErrorCompensator.DEFAULT_COMPENSATOR,
        "nr_feeds": ExposureErrorCompensator.DEFAULT_NR_FEEDS,
        "block_size": ExposureErrorCompensator.DEFAULT_BLOCK_SIZE,
        "finder": SeamFinder.DEFAULT_SEAM_FINDER,
        "final_megapix": Images.Resolution.FINAL.value,
        "blender_type": Blender.DEFAULT_BLENDER,
        "blend_strength": Blender.DEFAULT_BLEND_STRENGTH,
        "timelapse": Timelapser.DEFAULT_TIMELAPSE,
        "timelapse_prefix": Timelapser.DEFAULT_TIMELAPSE_PREFIX,
    }

    def __init__(self, **kwargs):
        self.time = time.time()
        self.initialize_stitcher(**kwargs)
        print("initialize_stitcher cost time:", self.cost_time())

    def cost_time(self):
        temp = time.time()
        cost = temp - self.time
        self.time = temp
        return cost

    def initialize_stitcher(self, **kwargs):
        self.settings = self.DEFAULT_SETTINGS.copy() #  先加载默认设置
        self.validate_kwargs(kwargs) # 验证参数，如果参数都是在default参数里面则通过，否则报错
        self.kwargs = kwargs # 保存参数
        self.settings.update(kwargs) #更新参数

        args = SimpleNamespace(**self.settings) # 将参数转换为对象
        self.medium_megapix = args.medium_megapix # 中等分辨率
        self.low_megapix = args.low_megapix # 低分辨率
        self.final_megapix = args.final_megapix # 最终分辨率
        if args.detector in ("orb", "sift"): # 如果检测器是orb或者sift
            self.detector = FeatureDetector(args.detector, nfeatures=0, nOctaveLayers=10, contrastThreshold=0.04, edgeThreshold=0, sigma=1.6, enable_precise_upscale=None)
            # self.detector = FeatureDetector(args.detector, nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6, enable_precise_upscale=None)
        else: # 否则使用默认的检测器
            self.detector = FeatureDetector(args.detector)
        match_conf = FeatureMatcher.get_match_conf(args.match_conf, args.detector)
        self.matcher = FeatureMatcher( # 特征匹配器，homography。
            args.matcher_type,
            args.range_width,
            try_use_gpu=args.try_use_gpu,
            # match_conf=match_conf,
        )
        self.subsetter = Subsetter(
            args.confidence_threshold, args.matches_graph_dot_file
        )
        self.camera_estimator = CameraEstimator(args.estimator) # 相机预测 homography,
        self.camera_adjuster = CameraAdjuster( # 相机调整
            args.adjuster, args.refinement_mask, args.confidence_threshold
        )
        self.wave_corrector = WaveCorrector(args.wave_correct_kind) # 波形校正 但是不需要
        self.warper = Warper(args.warper_type) # 图像扭曲
        self.warp_adjuster = WarperAdjuster() # 图像扭曲调整
        self.cropper = Cropper(args.crop) # 裁剪
        self.compensator = ExposureErrorCompensator( # 曝光补偿
            args.compensator, args.nr_feeds, args.block_size
        )
        self.seam_finder = SeamFinder(args.finder) # 接缝查找
        self.sensor_seam_finder = SensorSeamFinder()
        # 接缝产找以后，还可以绘制拼接缝
        self.blender = Blender(args.blender_type, args.blend_strength) # 混合
        self.timelapser = Timelapser(args.timelapse, args.timelapse_prefix) # 时间序列

    def stitch_verbose(self, images, feature_masks=[], verbose_dir=None): # 详细拼接
        return verbose_stitching(self, images, feature_masks, verbose_dir)

    def stitch(self, images, feature_masks=[] , road_masks_img = []):
        self.cracks = None
        self.origin = None
        self.images = Images.of(
            images, self.medium_megapix, self.low_megapix, self.final_megapix
        )
        # origin = road_masks_img
        if len(road_masks_img) > 0:
            self.origin = Images.of(road_masks_img, self.medium_megapix, self.low_megapix, self.final_megapix)

        imgs = self.resize_medium_resolution() # 中等分辨率 放缩到中等分辨率, 原来的图片还在
        # road_masks_img = self.resize_medium_resolution(origin)
        road_masks_img = self.origin.resize(Images.Resolution.MEDIUM)
        features = self.find_features(imgs, feature_masks) # 计算每个图片的特征点，进行了detect和compute， 这里 features_masks是个空
        # 这里的matches 是一个二维矩阵，记录两两对应的特征匹配情况
        matches = self.match_features(features) # 至此 特征提取，描述计算，单相应矩阵和内外点，以及优秀匹配特征点计算已完毕。
        # 在这步就将matches恢复成二维矩阵？
        imgs, features, matches, road_masks_img = self.subset(imgs, features, matches, road_masks_img) # 在这里已经得到了去除无关图片之后的缩略图片、特征、匹配特征矩阵，
        cameras = self.estimate_camera_parameters(features, matches) # 这里是得到了每张照片全局的相机变化参数R，是基于变换矩阵H得到的
        print("base cost time:", self.cost_time())
        cameras = self.refine_camera_parameters(features, matches, cameras) # 这里得到的cameras是每个图片的变换参数R，T，K，这里进行了相机参数的优化
        print("refine_camera_parameters cost time:", self.cost_time()) # 时间168.24370980262756
        cameras = self.perform_wave_correction(cameras) # 因为不进行波纹矫正，这个可以不要
        self.estimate_scale(cameras)
        # imgs = self.resize_low_resolution(imgs)
        # road_masks_img = self.resize_low_resolution(road_masks_img)
        # imgs, masks, corners, sizes = self.warp_low_resolution(imgs, cameras) # 这里进行了图像的扭曲，得到了扭曲后的图片，以及对应的mask，corners，sizes
        imgs, masks, corners, sizes = self.warp_medium_resolution(imgs, cameras) # 这里进行了图像的扭曲，得到了扭曲后的图片，以及对应的mask，corners，sizes
        road_masks_img, road_masks, road_corners, road_sizes = self.warp_medium_resolution(road_masks_img, cameras) # 这里进行了图像白线黄线的扭曲
        # road_masks_img, road_masks, road_corners, road_sizes = self.warp_low_resolution(road_masks_img, cameras) # 这里进行了图像白线黄线的扭曲
        # 使用final去调整
        # road_masks_img = origin.resize(Images.Resolution.FINAL)
        # imgs = self.resize_final_resolution()
        # imgs, masks, corners, sizes = self.warp_final_resolution(imgs, cameras)
        # road_masks_img, road_masks, road_corners, road_sizes = self.warp_final_resolution(road_masks_img, cameras)
        ##################################################
        # corners = self.warp_adjuster.adjust(imgs, corners, road_masks_img)
        ##################################################
        # self.prepare_cropper(imgs, masks, corners, sizes) # 因为没有进行 crop 所以这里没有进行裁剪, 这部可以不执行
        # imgs, masks, corners, sizes = self.crop_low_resolution( # 因为没有进行 crop 所以这里没有进行裁剪, 这部可以不执行
        #     imgs, masks, corners, sizes
        # )
        #************************************************
        '''要是去除车辆，则需要在这里对warp过的图片进行处理，从相邻图片里面的重叠区域取像素'''
        # ************************************************

        # self.estimate_exposure_errors(corners, imgs, masks)
        seam_masks = self.find_seam_masks(imgs, corners, masks) # 从这里修改拼接缝 masks是白色的掩膜.如果使用原图进行接缝查找，会导致内存越界，报错结果：error: (-215:Assertion failed) u != 0 in function 'cv::UMat::create'
        # self.seam_finder
        # test_seam_masks = self.sensor_seam_finder.find(imgs, corners, testMasks)
        # print(test_seam_masks)



        imgs = self.resize_final_resolution()
        imgs, masks, corners, sizes = self.warp_final_resolution(imgs, cameras)
        # imgs, masks, corners, sizes = self.crop_final_resolution(
        #     imgs, masks, corners, sizes
        # )

        self.set_masks(masks)
        # 应该是这里计算出来errors的原因吧！！！！！！！！ 设置的为NO，应该可以跳过
        # imgs = self.compensate_exposure_errors(corners, imgs)
        seam_masks = self.resize_seam_masks(seam_masks)

        ############################################
        # 实际调整应该在这里调整corners, 在上面调整只会影响seam-cutting的查找吧！
        if self.origin:
            road_masks_img = self.origin.resize(Images.Resolution.FINAL)
            road_masks_img, road_masks, road_corners, road_sizes = self.warp_final_resolution(road_masks_img, cameras)
            # corners = self.warp_adjuster.adjust(imgs, corners, road_masks_img, seam_masks)
            corners = self.warp_adjuster.adjust(imgs, corners, road_masks_img, seam_masks)
        #############################################



        self.initialize_composition(corners, sizes)
        self.blend_images(imgs, seam_masks, corners)
        panorama = self.create_final_panorama()


        #################################
        # 生成中央黄线的代码
        # road_masks_img = self.origin.resize(Images.Resolution.FINAL)
        # road_masks_img, road_masks, road_corners, road_sizes = self.warp_final_resolution(road_masks_img, cameras)
        # # road_imgs = self.compensate_exposure_errors(corners, road_imgs)
        # self.initialize_composition(corners, sizes)
        # self.blend_images(road_masks_img, seam_masks, corners)
        # road_panorama = self.create_final_panorama()
        # cv2.imwrite("./roadmask.jpg", road_panorama)
        # print("create road mask panorama cost time:", self.cost_time())  # 花费时间

        ################################

        blend_seam_masks = SeamFinder.blend_seam_masks(seam_masks, corners, sizes)
        cv2.imwrite("stitch10_blend_seam.jpg", blend_seam_masks)
        print("create_final_panorama cost time:", self.cost_time()) # 花费时间 80秒
        panorama_with_seam_lines = SeamFinder.draw_seam_lines(panorama, blend_seam_masks, 3)
        print("draw_seam_lines cost time:", self.cost_time()) # 花费时间
        with_seam_polygons = SeamFinder.draw_seam_polygons(panorama, blend_seam_masks)
        print("draw_seam_polygons cost time:", self.cost_time()) # 花费时间
        return panorama, panorama_with_seam_lines, with_seam_polygons

    def stitch_with_camera_data(self):
        '''加载cameras数据，前面没有save过滤掉数据文件，所以这里没有再特数据处理，只做实验而用'''

        import json
        with open("./cameras.json", "r") as f:
            data = json.load(f)
        cameras = self.json_to_cameras(data["cameras"])
        imgs = data["images"]
        road_masks_img = data["origin"]
        cracks = data["cracks"]
        self.estimate_scale(cameras)
        self.images = Images.of(
            imgs, self.medium_megapix, self.low_megapix, self.final_megapix
        )  # 这里images是warp后的图片
        self.origin = Images.of(road_masks_img, self.medium_megapix, self.low_megapix, self.final_megapix)
        if len(cracks) == 0:
            self.cracks = None
        else:
            self.cracks = Images.of(cracks, self.medium_megapix, self.low_megapix, self.final_megapix)
        imgs = self.resize_medium_resolution()  # 中等分辨率 放缩到中等分辨率, 原来的图片还在
        road_masks_img = self.resize_medium_resolution(self.origin)
        imgs, masks, corners, sizes = self.warp_medium_resolution(imgs,cameras)  # 这里进行了图像的扭曲，得到了扭曲后的图片，以及对应的mask，corners，sizes
        road_masks_img, road_masks, road_corners, road_sizes = self.warp_medium_resolution(road_masks_img,cameras)  # 这里进行了图像白线黄线的扭曲
        seam_masks = self.find_seam_masks(imgs, corners, masks) # 从这里修改拼接缝 masks是白色的掩膜.如果使用原图进行接缝查找，会导致内存越界，报错结果：error: (-215:Assertion failed) u != 0 in function 'cv::UMat::create'
        imgs = self.resize_final_resolution()
        imgs, masks, corners, sizes = self.warp_final_resolution(imgs, cameras)
        self.set_masks(masks)
        seam_masks = self.resize_seam_masks(seam_masks)

        ############################################
        # 实际调整应该在这里调整corners, 在上面调整只会影响seam-cutting的查找吧！
        road_masks_img = self.origin.resize(Images.Resolution.FINAL)
        road_masks_img, road_masks, road_corners, road_sizes = self.warp_final_resolution(road_masks_img, cameras)
        # corners = self.warp_adjuster.adjust(imgs, corners, road_masks_img, seam_masks)
        corners = self.warp_adjuster.adjust(imgs, corners, road_masks_img, seam_masks)
        #############################################

        self.initialize_composition(corners, sizes)
        self.blend_images(imgs, seam_masks, corners)
        panorama = self.create_final_panorama()
        print("create_final_panorama cost time:", self.cost_time())
        #################################
        # 生成中央黄线的代码
        # self.prepare_cropper(road_masks_img, road_masks, road_corners, road_sizes)  # 因为没有进行 crop 所以这里没有进行裁剪, 这部可以不执行
        # road_masks_img, road_masks, road_corners, road_sizes = self.crop_low_resolution(  # 因为没有进行 crop 所以这里没有进行裁剪, 这部可以不执行
        #     road_masks_img, road_masks, road_corners, road_sizes
        # )
        #
        # self.estimate_exposure_errors(road_corners, road_masks_img, road_masks)
        # road_imgs = origin.resize(Images.Resolution.FINAL)
        road_masks_img = self.origin.resize(Images.Resolution.FINAL)
        road_masks_img, road_masks, road_corners, road_sizes = self.warp_final_resolution(road_masks_img, cameras)
        # road_imgs = self.compensate_exposure_errors(corners, road_imgs)
        self.initialize_composition(corners, sizes)
        self.blend_images(road_masks_img, seam_masks, corners)
        road_panorama = self.create_final_panorama()
        cv2.imwrite("./roadmask.jpg", road_panorama)
        print("create road mask panorama cost time:", self.cost_time())  # 花费时间

        ################################
        ################################
        '''生成crack'''
        if self.cracks:
            crack_img = self.crack.resize(Images.Resolution.FINAL)
            crack_img, crack_masks, crack_corners, crack_sizes = self.warp_final_resolution(crack_img, cameras)
            self.initialize_composition(corners, sizes)
            self.blend_images(crack_img, seam_masks, corners)
            crack_panorama = self.create_final_panorama()
            cv2.imwrite("./crack.jpg", crack_panorama)
            print("create crack panorama cost time:", self.cost_time())  # 花费时间
        #################################

        blend_seam_masks = SeamFinder.blend_seam_masks(seam_masks, corners, sizes)
        cv2.imwrite("stitch10_blend_seam.jpg", blend_seam_masks)
        print("create_final_panorama cost time:", self.cost_time())  # 花费时间 80秒
        panorama_with_seam_lines = SeamFinder.draw_seam_lines(panorama, blend_seam_masks, 3)
        print("draw_seam_lines cost time:", self.cost_time())  # 花费时间
        with_seam_polygons = SeamFinder.draw_seam_polygons(panorama, blend_seam_masks)
        print("draw_seam_polygons cost time:", self.cost_time())  # 花费时间
        return panorama, panorama_with_seam_lines, with_seam_polygons
        pass

    def produce_stitch_cameras_data(self, images, feature_masks=[] , road_masks_img = [], crack_imgs = []):
        '''只是为了产生cameras数据，为了节省实验时的拼接时间'''
        self.crack = None
        self.images = Images.of(
            images, self.medium_megapix, self.low_megapix, self.final_megapix
        )
        # origin = road_masks_img
        if len(crack_imgs) != 0:
            self.crack = Images.of(crack_imgs, self.medium_megapix, self.low_megapix, self.final_megapix)
        origin = Images.of(road_masks_img, self.medium_megapix, self.low_megapix, self.final_megapix)
        self.origin = origin
        imgs = self.resize_medium_resolution()  # 中等分辨率 放缩到中等分辨率, 原来的图片还在
        road_masks_img = self.origin.resize(Images.Resolution.MEDIUM)
        # road_masks_img = self.resize_medium_resolution(origin)

        if self.crack:
            crack_imgs = self.crack.resize(Images.Resolution.MEDIUM)

        features = self.find_features(imgs, feature_masks)  # 计算每个图片的特征点，进行了detect和compute， 这里 features_masks是个空
        # 这里的matches 是一个二维矩阵，记录两两对应的特征匹配情况
        matches = self.match_features(features)  # 至此 特征提取，描述计算，单相应矩阵和内外点，以及优秀匹配特征点计算已完毕。
        # 在这步就将matches恢复成二维矩阵？
        imgs, features, matches, road_masks_img = self.subset(imgs, features, matches,
                                                              road_masks_img)  # 在这里已经得到了去除无关图片之后的缩略图片、特征、匹配特征矩阵，
        cameras = self.estimate_camera_parameters(features, matches)  # 这里是得到了每张照片全局的相机变化参数R，是基于变换矩阵H得到的
        print("base cost time:", self.cost_time())
        cameras = self.refine_camera_parameters(features, matches, cameras)  # 这里得到的cameras是每个图片的变换参数R，T，K，这里进行了相机参数的优化
        print("refine_camera_parameters cost time:", self.cost_time())  # 时间168.24370980262756
        cameras = self.perform_wave_correction(cameras)  # 因为不进行波纹矫正，这个可以不要
        self.estimate_scale(cameras)
        data = {}
        data["cameras"] = self.cameras_to_json(cameras)
        data["images"] = self.images.getImgs()
        data["origin"] = self.origin.getImgs()
        data["cracks"] = self.crack.getImgs() if self.crack else []
        import json
        with open("./cameras.json", "w") as f:
            json.dump(data, f)

    def cameras_to_json(self, cameras):
        data = []
        for camera in cameras:
            temp = {}
            temp["R"]  = camera.R.tolist()
            temp["aspect"] = camera.aspect
            temp["focal"] = camera.focal
            temp["ppx"] = camera.ppx
            temp["ppy"] = camera.ppy
            temp["t"] = camera.t.tolist()
            data.append(temp)
        return data

    def json_to_cameras(self, cameras):
        data = []
        for camera in cameras:
            temp = cv2.detail.CameraParams()
            temp.R = np.array(camera["R"], dtype=np.float32)
            temp.aspect = camera["aspect"]
            temp.focal = camera["focal"]
            temp.ppx = camera["ppx"]
            temp.ppy = camera["ppy"]
            temp.t = np.array(camera["t"], dtype=np.float32)
            data.append(temp)

        return tuple(data)
    def write_verbose_result(self, dir_name, img_name, img):
        cv2.imwrite(self.verbose_output(dir_name, img_name), img)

    def verbose_output(self, dir_name, file):
        return os.path.join(dir_name, file)
    def resize_medium_resolution(self, imgs = None):
        return list(self.images.resize(Images.Resolution.MEDIUM, imgs))

    def find_features(self, imgs, feature_masks=[]):
        if len(feature_masks) == 0:
            return self.detector.detect(imgs)
        else:
            feature_masks = Images.of(
                feature_masks, self.medium_megapix, self.low_megapix, self.final_megapix
            )
            feature_masks = list(feature_masks.resize(Images.Resolution.MEDIUM))
            feature_masks = [Images.to_binary(mask) for mask in feature_masks]
            return self.detector.detect_with_masks(imgs, feature_masks)

    def match_features(self, features):
        return self.matcher.match_features(features)

    # 将拼接时用不到的图片，特征，匹配点都去除
    def subset(self, imgs, features, matches, road_masks):
        indices = self.subsetter.subset(self.images.names, features, matches)
        imgs = Subsetter.subset_list(imgs, indices) # 根据索引提取图片，去除不链接的图片之后的图片集合
        road_masks = Subsetter.subset_list(road_masks, indices)
        features = Subsetter.subset_list(features, indices)

        matches = Subsetter.subset_matches(matches, indices)

        #将原来的真实图片也去除无关的图片。
        self.images.subset(indices)
        if self.origin:
            self.origin.subset(indices)
        if self.crack:
            self.crack.subset(indices)
        return imgs, features, matches, road_masks

    def estimate_camera_parameters(self, features, matches):
        return self.camera_estimator.estimate(features, matches)

    def refine_camera_parameters(self, features, matches, cameras):
        return self.camera_adjuster.adjust(features, matches, cameras)

    def perform_wave_correction(self, cameras):
        return self.wave_corrector.correct(cameras)

    def estimate_scale(self, cameras):
        self.warper.set_scale(cameras)

    def resize_low_resolution(self, imgs=None):
        return list(self.images.resize(Images.Resolution.LOW, imgs))

    def warp_low_resolution(self, imgs, cameras):
        sizes = self.images.get_scaled_img_sizes(Images.Resolution.LOW)
        camera_aspect = self.images.get_ratio(
            Images.Resolution.MEDIUM, Images.Resolution.LOW
        )
        imgs, masks, corners, sizes = self.warp(imgs, cameras, sizes, camera_aspect)
        return list(imgs), list(masks), corners, sizes

    def warp_medium_resolution(self, imgs, cameras):
        sizes = self.images.get_scaled_img_sizes(Images.Resolution.MEDIUM)
        camera_aspect = self.images.get_ratio(
            Images.Resolution.MEDIUM, Images.Resolution.MEDIUM
        )
        imgs, masks, corners, sizes = self.warp(imgs, cameras, sizes, camera_aspect)
        return list(imgs), list(masks), corners, sizes

    def warp_final_resolution(self, imgs, cameras):
        sizes = self.images.get_scaled_img_sizes(Images.Resolution.FINAL)
        camera_aspect = self.images.get_ratio(
            Images.Resolution.MEDIUM, Images.Resolution.FINAL
        )
        return self.warp(imgs, cameras, sizes, camera_aspect)

    def warp(self, imgs, cameras, sizes, aspect=1):
        imgs = self.warper.warp_images(imgs, cameras, aspect)
        masks = self.warper.create_and_warp_masks(sizes, cameras, aspect)
        corners, sizes = self.warper.warp_rois(sizes, cameras, aspect)
        return imgs, masks, corners, sizes

    def prepare_cropper(self, imgs, masks, corners, sizes):
        self.cropper.prepare(imgs, masks, corners, sizes)

    def crop_low_resolution(self, imgs, masks, corners, sizes):
        imgs, masks, corners, sizes = self.crop(imgs, masks, corners, sizes)
        return list(imgs), list(masks), corners, sizes

    def crop_final_resolution(self, imgs, masks, corners, sizes):
        lir_aspect = self.images.get_ratio(
            Images.Resolution.LOW, Images.Resolution.FINAL
        )
        return self.crop(imgs, masks, corners, sizes, lir_aspect)

    def crop(self, imgs, masks, corners, sizes, aspect=1):
        masks = self.cropper.crop_images(masks, aspect)
        imgs = self.cropper.crop_images(imgs, aspect)
        corners, sizes = self.cropper.crop_rois(corners, sizes, aspect)
        return imgs, masks, corners, sizes

    def estimate_exposure_errors(self, corners, imgs, masks):
        self.compensator.feed(corners, imgs, masks)

    def find_seam_masks(self, imgs, corners, masks):
        return self.seam_finder.find(imgs, corners, masks)

    def resize_final_resolution(self):
        return self.images.resize(Images.Resolution.FINAL)

    def compensate_exposure_errors(self, corners, imgs):
        compensated_imgs = []
        for idx, (corner, img) in enumerate(zip(corners, imgs)):
            mask = self.get_mask(idx)
            img = self.compensator.apply(idx, corner, img, mask)
            compensated_imgs.append(img)
        return compensated_imgs
        # for idx, (corner, img) in enumerate(zip(corners, imgs)):
        #     yield self.compensator.apply(idx, corner, img, self.get_mask(idx))

    def resize_seam_masks(self, seam_masks):
        masks = []
        for idx, seam_mask in enumerate(seam_masks):
            mask = SeamFinder.resize(seam_mask, self.get_mask(idx))
            masks.append(mask)
        return masks
        # for idx, seam_mask in enumerate(seam_masks):
        #     yield SeamFinder.resize(seam_mask, self.get_mask(idx))

    def set_masks(self, mask_generator):
        self.masks = mask_generator
        # self.mask_index = -1

    def get_mask(self, idx):
        return self.masks[idx]
        if idx == self.mask_index + 1:
            self.mask_index += 1
            self.mask = next(self.masks)
            return self.mask
        elif idx == self.mask_index:
            return self.mask
        else:
            raise StitchingError("Invalid Mask Index!")

    def initialize_composition(self, corners, sizes):
        if self.timelapser.do_timelapse:
            self.timelapser.initialize(corners, sizes)
        else:
            self.blender.prepare(corners, sizes)

    def blend_images(self, imgs, masks, corners):
        for idx, (img, mask, corner) in enumerate(zip(imgs, masks, corners)):
            if self.timelapser.do_timelapse:
                self.timelapser.process_and_save_frame(
                    self.images.names[idx], img, corner
                )
            else:
                self.blender.feed(img, mask, corner)

    def create_final_panorama(self):
        if not self.timelapser.do_timelapse:
            panorama, _ = self.blender.blend()
            return panorama

    def validate_kwargs(self, kwargs):
        for arg in kwargs:
            if arg not in self.DEFAULT_SETTINGS:
                raise StitchingError("Invalid Argument: " + arg)


class AffineStitcher(Stitcher):
    AFFINE_DEFAULTS = {
        "estimator": "affine",
        "wave_correct_kind": "no",
        "matcher_type": "affine",
        "adjuster": "affine",
        "warper_type": "affine",
        "compensator": "no",
    }

    DEFAULT_SETTINGS = Stitcher.DEFAULT_SETTINGS.copy()
    DEFAULT_SETTINGS.update(AFFINE_DEFAULTS)

    def initialize_stitcher(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.AFFINE_DEFAULTS and value != self.AFFINE_DEFAULTS[key]:
                warnings.warn(
                    f"You are overwriting an affine default ({key}={self.AFFINE_DEFAULTS[key]}) with another value ({value}). Make sure this is intended",  # noqa: E501
                    StitchingWarning,
                )
        super().initialize_stitcher(**kwargs)
