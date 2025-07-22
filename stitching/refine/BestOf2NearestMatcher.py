import cv2
import numpy as np

class BestOf2NearestMatcher:
    def __init__(self, ratio_thresh=0.8, ransac_thresh=3.0):
        self.ratio_thresh = ratio_thresh  # 比率测试阈值
        self.ransac_thresh = ransac_thresh  # RANSAC 阈值


    def apply2(self, features):
        """
        可以尝试使用多线程进行处理：https://blog.csdn.net/zhangphil/article/details/88577091
        import concurrent.futures
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
        futures = [executor.submit(task, f'Task {i}') for i in range(10)]
        concurrent.futures.wait(futures) # 等待所有任务完成

        对输入的特征进行匹配

        # 只要good——matches数据量够少，那么计算refine——camera时就会更快
        :param features: 包含图像特征的列表，每个元素是 cv2.detail.ImageFeatures 对象
        :return: 匹配结果列表，每个元素是 MatchesInfo 对象
        """
        pairwise_matches = []
        matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
        wdith, height = features[0].img_size
        # 遍历所有图像对
        for i in range(len(features)):
            for j in range(len(features)):
                if i == j:
                    matches_info = cv2.detail.MatchesInfo()
                    matches_info.src_img_idx = -1
                    matches_info.dst_img_idx = -1
                    matches_info.matches = []
                    matches_info.num_inliers = 0
                    matches_info.H = None
                    matches_info.confidence = 0.0
                    matches_info.inliers_mask = None
                    pairwise_matches.append(matches_info)
                    continue
                if abs(j-i)>=2:
                    # 存储匹配结果
                    matches_info = cv2.detail.MatchesInfo(matches_info)
                    matches_info.src_img_idx = i
                    matches_info.dst_img_idx = j
                    matches_info.matches = []
                    matches_info.num_inliers = 0
                    matches_info.H = None
                    matches_info.confidence = 0.0
                    matches_info.inliers_mask = None

                    pairwise_matches.append(matches_info)
                    continue
                descriptors1 = features[i].descriptors
                descriptors2 = features[j].descriptors

                # 最近邻匹配（k=2）
                knn_matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

                # 比率测试筛选匹配点
                good_matches = []
                for m, n in knn_matches:
                    if m.distance < self.ratio_thresh * n.distance:
                        good_matches.append(m)

                #把goodmatches里面超过高度 60%的剔除掉，距离差
                # threshold = height * 0.4
                newgoodmatches = []
                kps1 = features[i].keypoints
                kps2 = features[j].keypoints

                distance = 0
                for m in good_matches:
                    original = kps1[m.queryIdx]
                    transform = kps2[m.trainIdx]
                    pt1 = original.pt  # x,y
                    pt2 = transform.pt
                    distance += abs(pt1[1] - pt2[1])
                    # distance += abs((pt1[1] - pt2[1])/(pt1[0]-pt2[0]))
                distance_everage = distance / len(good_matches)
                threshold = distance_everage * 0.1
                for m in good_matches:
                    original = kps1[m.queryIdx]
                    transform = kps2[m.trainIdx]
                    pt1 = original.pt  # x,y
                    pt2 = transform.pt

                    if abs(abs(pt1[1]-pt2[1]) -distance_everage) < threshold and abs(pt1[0] - pt2[0]) < 100:
                    # if abs((pt1[1]-pt2[1])/(pt1[0]-pt2[0]) -distance_everage) < threshold and abs(pt1[0] - pt2[0]) < 100:
                        newgoodmatches.append(m)
                good_matches = newgoodmatches

                # 后续再找到对应的features里面匹配的点，只留下good_matches匹配点， 然后对于更新每个matches_info里面的matches向量


                # 几何验证（使用 RANSAC 计算单应性矩阵）
                H = None
                inliers_mask = []
                if len(good_matches) >= 4:
                    src_points = np.float32([features[i].keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
                    dst_points = np.float32([features[j].keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
                    H, inliers_mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, self.ransac_thresh)



                matches_info = cv2.detail.MatchesInfo(matches_info)
                matches_info.src_img_idx = i
                matches_info.dst_img_idx = j
                matches_info.matches = good_matches
                matches_info.num_inliers = np.sum(inliers_mask)
                matches_info.H = H
                matches_info.confidence = matches_info.num_inliers / (8+0.3*len(good_matches)) if good_matches else 0.0
                matches_info.inliers_mask = inliers_mask.flatten() if inliers_mask.any() else ()
                pairwise_matches.append(matches_info)

        return pairwise_matches
    def collectGarbage(self):
        pass