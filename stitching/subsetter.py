import warnings
from itertools import chain

import cv2 as cv
import numpy as np

from .feature_matcher import FeatureMatcher
from .stitching_error import StitchingError, StitchingWarning
import math

import platform

class Subsetter:
    """https://docs.opencv.org/4.x/d7/d74/group__stitching__rotation.html#ga855d2fccbcfc3b3477b34d415be5e786 and
    https://docs.opencv.org/4.x/d7/d74/group__stitching__rotation.html#gabaeb9dab170ea8066ae2583bf3a669e9
    """  # noqa

    DEFAULT_CONFIDENCE_THRESHOLD = 1
    DEFAULT_MATCHES_GRAPH_DOT_FILE = None

    def __init__(
        self,
            # 这里我设置的是0.995， 默认是1
        confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
        matches_graph_dot_file=DEFAULT_MATCHES_GRAPH_DOT_FILE,
    ):
        self.confidence_threshold = confidence_threshold
        self.save_file = matches_graph_dot_file

    def subset(self, img_names, features, matches):
        self.save_matches_graph_dot_file(img_names, matches)
        indices = self.get_indices_to_keep(features, matches)

        if len(indices) < len(img_names):
            warnings.warn(
                "Not all images are included in the final panorama. If this is not intended, use the 'matches_graph_dot_file' parameter to analyze your matches. You might want to lower the 'confidence_threshold' or try another 'detector' to include all your images.",  # noqa: E501
                StitchingWarning,
            )

        return indices



    def save_matches_graph_dot_file(self, img_names, pairwise_matches):
        if self.save_file:
            with open(self.save_file, "w") as filehandler:
                filehandler.write(self.get_matches_graph(img_names, pairwise_matches))

    def get_matches_graph(self, img_names, pairwise_matches):
        return cv.detail.matchesGraphAsString(
            img_names,
            pairwise_matches,
            (
                0.00001  # see issue #56
                if (self.confidence_threshold == 0)
                else self.confidence_threshold
            ),
        )


    def get_indices_to_keep(self, features, pairwise_matches):
        # indices = self.leave_biggest_component(
        indices = cv.detail.leaveBiggestComponent(
            features, pairwise_matches, self.confidence_threshold
        )

        # see https://github.com/OpenStitching/stitching/issues/40
        # indices = indices.flatten() # 转为一维数组

        if len(indices) < 2:
            raise StitchingError(
                "No match exceeds the given confidence threshold. Do your images have enough overlap and common features? If yes, you might want to lower the 'confidence_threshold' or try another 'detector'."  # noqa: E501
            )

        return indices

    @staticmethod
    def subset_list(list_to_subset, indices):
        return [list_to_subset[i] for i in indices]

    @staticmethod
    def subset_matches(pairwise_matches, indices):
        # 原来的pairwise_matches是个二维矩阵，但是现在肯定不是二维矩阵了， 需要重写一下
        # 现在肯定是需要第一张和最后一张都必须包含，那么根据index就可以获取了吧？

        # 先得到原来的特征匹配对矩阵
        matches_matrix = FeatureMatcher.get_matches_matrix(pairwise_matches)
        # 然后根据索引得到新的特征匹配对矩阵，去除不需要的特征匹配
        matches_matrix_subset = matches_matrix[np.ix_(indices, indices)]
        # 最后将新的特征匹配对矩阵转换为列表
        matches_subset_list = list(chain.from_iterable(matches_matrix_subset.tolist()))
        return matches_subset_list

    def is_windows(self):
        os_name = platform.system()

        if os_name.lower() == 'windows':
            return True
        elif os_name.lower() == 'linux':
            return False
        else:
            False
