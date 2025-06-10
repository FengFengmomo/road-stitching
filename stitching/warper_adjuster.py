import cv2
import numpy as np


class WarperAdjuster:
    def __init__(self):
        pass


    def adjust(self, imgs, corners, road_masks_imgs, masks):
        '''
        这里的mask是公路白线掩码， 和images一样都是warp以后的
        '''
        corners = [list(ele) for ele in corners]
        masks = [mask.get() for mask in masks]
        for i in range(len(imgs)-1):
            rang1, range2 = self.get_img_range(imgs[i], corners[i]), self.get_img_range(imgs[i+1], corners[i+1])
            overlap = self.get_overlap(rang1, range2)
            road_mask_roi1, road_mask_roi2 = self.get_overlap_regions_pixels(road_masks_imgs[i], corners[i], road_masks_imgs[i + 1], corners[i + 1], overlap)
            road_mask_roi1, road_mask_roi2 = cv2.cvtColor(road_mask_roi1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(road_mask_roi2, cv2.COLOR_BGR2GRAY)
            mask_roi1, mask_roi2 = self.get_overlap_regions_pixels(masks[i], corners[i], masks[i + 1], corners[i + 1], overlap)
            # index1, index2 = self.get_mask_intersect_min_index(road_mask_roi1, mask_roi1, True), self.get_mask_intersect_min_index(road_mask_roi2, mask_roi2, False)
            index1, index2 = self.get_mask_intersect_min_index(road_mask_roi1, mask_roi1, road_mask_roi2, mask_roi2)
            # index1 = self.get_mask_index(road_mask_roi1)
            # index2 = self.get_mask_index(road_mask_roi2)
            # # margin = index2[2] - index1[2] + index2[0] - index1[0] + index2[1] - index1[1]
            # margin = index2[1] - index1[1]
            # margin = int(margin/2) # 两个掩码的索引差值
            # margin = int(index2[1] - index1[1]) # 两个掩码的索引差值
            # point1 = overlap[0] - corners[i][0] + index1[0], overlap[1] + corners[i][1]+index1[1]
            # point2 = overlap[0] - corners[i+1][0] + index2[0], overlap[1] + corners[i+1][1]+index2[1]
            # cv2.circle(imgs[i], point1, 30, (255, 0, 0), -1)
            # cv2.circle(imgs[i+1], point2, 30, (0, 0, 255), -1)
            margin = index2 - index1

            # if i+1 == 8:
            #     margin+=12
            # if i+1 == 9:
            #     margin -=5
            print("index1:", index1, " index2:", index2, f' {i+1} 相对{i} 偏移了 {-margin}')
            if margin == 0:
                continue
            # print(f' {i+1} 相对{i} 偏移了 {margin}')
            # margin大于零说明下一张图片偏右了， 否则就是偏左了
            for j in range(i+1,len(imgs)):
                corners[j][0] -= margin

        corners = [tuple(ele) for ele in corners]

        return corners

    def get_image_ranges(self, imgs, corners):
        """
        计算每个图片的坐标范围 (x_min, y_min) 到 (x_max, y_max)
        """
        ranges = []
        for (x, y), img in zip(corners,imgs):
            height, width = img.shape[0], img.shape[1] # 行数rows height，列数cols width. 高，宽, 要转换为灰度图吗？
            x_min = x
            y_min = y
            x_max = x + width
            y_max = y + height
            ranges.append(((x_min, y_min), (x_max, y_max)))
        return ranges

    def get_img_range(self, img, corner):
        """
        计算单个图片的坐标范围 (x_min, y_min) 到 (x_max, y_max)
        """
        height, width = img.shape[0], img.shape[1] # 行数rows height，列数cols width. 高，宽, 要转换为灰度图吗？
        x_min = corner[0]
        y_min = corner[1]
        x_max = corner[0] + width
        y_max = corner[1] + height
        return ((x_min, y_min), (x_max, y_max))

    def get_overlap(self, range1, range2):
        """
        输入两个图片的坐标范围，返回重叠区域的范围 (x1, y1, x2, y2)
        若无重叠返回None
        """
        # 解包坐标范围
        (x1_min, y1_min), (x1_max, y1_max) = range1
        (x2_min, y2_min), (x2_max, y2_max) = range2

        # 计算X轴重叠
        overlap_x_min = max(x1_min, x2_min)
        overlap_x_max = min(x1_max, x2_max)

        # 计算Y轴重叠
        overlap_y_min = max(y1_min, y2_min)
        overlap_y_max = min(y1_max, y2_max)

        # 检查是否有有效重叠
        if overlap_x_min >= overlap_x_max or overlap_y_min >= overlap_y_max:
            return None

        return (overlap_x_min, overlap_y_min, overlap_x_max, overlap_y_max)

    def get_overlap_regions_pixels(self, img1, corner1, img2, corner2, overlap_global):
        """
        输入:
            img1, img2: 原始图像（numpy数组）
            corner1, corner2: 两图的全局左上角坐标 (x, y)
            overlap_global: 全局坐标系下的重叠区域 (x_min, y_min, x_max, y_max)
        输出:
            img1_roi, img2_roi: 两图在重叠区域的局部像素块（尺寸相同）
        """
        # 解包全局重叠区域坐标
        x_min_global, y_min_global, x_max_global, y_max_global = overlap_global

        # --------------------------------------------
        # 步骤1：将全局坐标转换为局部坐标
        # --------------------------------------------
        # 计算img1的局部坐标范围
        x1_min = x_min_global - corner1[0]
        y1_min = y_min_global - corner1[1]
        x1_max = x_max_global - corner1[0]
        y1_max = y_max_global - corner1[1]

        # 计算img2的局部坐标范围
        x2_min = x_min_global - corner2[0]
        y2_min = y_min_global - corner2[1]
        x2_max = x_max_global - corner2[0]
        y2_max = y_max_global - corner2[1]

        # --------------------------------------------
        # 步骤2：边界检查与截取
        # --------------------------------------------
        # 确保坐标在图像范围内
        def clamp_coordinates(img, x_min, y_min, x_max, y_max):
            h, w = img.shape[:2]
            x_start = max(0, int(round(x_min)))
            y_start = max(0, int(round(y_min)))
            x_end = min(w, int(round(x_max)))
            y_end = min(h, int(round(y_max)))
            return x_start, y_start, x_end, y_end

        # 获取img1的有效区域
        x1_start, y1_start, x1_end, y1_end = clamp_coordinates(img1, x1_min, y1_min, x1_max, y1_max)
        # 获取img2的有效区域
        x2_start, y2_start, x2_end, y2_end = clamp_coordinates(img2, x2_min, y2_min, x2_max, y2_max)

        # --------------------------------------------
        # 步骤3：提取像素区域
        # --------------------------------------------
        # 提取img1的重叠区域（注意OpenCV的索引是[y, x]）
        img1_roi = img1[y1_start:y1_end, x1_start:x1_end]
        # 提取img2的重叠区域
        img2_roi = img2[y2_start:y2_end, x2_start:x2_end]

        # --------------------------------------------
        # 步骤4：确保尺寸一致（可能因边界截断导致不一致）
        # --------------------------------------------
        # 取最小尺寸
        # min_height = min(img1_roi.shape[0], img2_roi.shape[0])
        # min_width = min(img1_roi.shape[1], img2_roi.shape[1])

        # 统一尺寸
        # img1_roi = img1_roi[:min_height, :min_width]
        # img2_roi = img2_roi[:min_height, :min_width]

        return img1_roi, img2_roi

    def get_mask_index(self, mask, padding = 30):
        width, height = mask.shape[1], mask.shape[0]
        mid = int(height / 2)
        index_0 = np.min(np.where(mask[mid+50] >50))
        index_h = np.min(np.where(mask[mid - 50] >50))
        index_mid = np.where(mask[mid] >50)
        index_low = np.where(mask[mid - padding] >50)
        index_high = np.where(mask[mid + padding] > 50)
        index_low_min = np.min(index_low)
        index_high_min = np.min(index_high)
        index_mid_min = np.min(index_mid)
        # 求三个数得平均
        return [index_low_min, index_mid_min, index_high_min, index_0, index_h]

    def get_mask_intersect_min_index(self, road_mask1, seam_mask1, road_mask2, seam_mask2):
        # 找到拼接缝和白线掩码相交的最左上角

        # seam_mask1[seam_mask1!=0] = 255
        # kernel = np.ones((3, 3), np.uint8)  # 定义膨胀核（调整核大小可控制边缘宽度）
        # dilated = cv2.dilate(seam_mask1, kernel, iterations=1)  # 膨胀
        # seam_mask1 = dilated - seam_mask1  # 获得边缘

        road_mask1[road_mask1 != 0] = 1
        seam_mask1[seam_mask1 != 0] = 1
        mask1 = road_mask1 * seam_mask1

        # seam_mask2[seam_mask2 != 0] = 255
        # kernel = np.ones((3, 3), np.uint8)  # 定义膨胀核（调整核大小可控制边缘宽度）
        # dilated = cv2.dilate(seam_mask2, kernel, iterations=1)  # 膨胀
        # seam_mask2 = dilated - seam_mask2  # 获得边缘
        road_mask2[road_mask2 != 0] = 2
        seam_mask2[seam_mask2 != 0] = 2
        mask2 = road_mask2 * seam_mask2

        mask = mask1 + mask2
        # mask[mask == 5] = 0

        mask[mask != 5] = 0 # 将非拼接缝部分设为0， 只剩下了非零区域

        rows, cols = np.nonzero(mask)
        if rows.size == 0:
            # 未检测到非零值，不进行偏移矫正
            return 0, 0
        # 计算行和列的边界
        r_min, r_max = rows.min(), rows.max()
        # c_min, c_max = cols.min(), cols.max()

        # 从两个区域计算两个区域的中间部分，核心位置，然后计算两个的偏移
        # 两个掩码，只取最小行和最大行之间的部分， 取全部列
        mask1_area = mask1[r_min: r_max + 1, :]
        mask2_area = mask2[r_min: r_max + 1, :]
        padding = int((r_max-r_min)/2)

        eves1 = []
        # 计算掩码非零部分的最大列索引和最小列索引
        for i in range(r_min, r_max):
            notzero = np.where(road_mask1[i] != 0)
            if len(notzero[0]) == 0:
                continue
            min_col = np.min(notzero)
            max_col = np.max(notzero)
            everage = (min_col + max_col)/2
            eves1.append(everage)
        eve1 = np.mean(np.array(eves1))
        eve1 = int(eve1)

        eves2 = []
        # 计算掩码非零部分的最大列索引和最小列索引
        for i in range(r_min, r_max):
            notzero = np.where(road_mask2[i] != 0)
            if len(notzero[0]) == 0:
                continue
            min_col = np.min(notzero)
            max_col = np.max(notzero)
            everage = (min_col + max_col)/2
            eves2.append(everage)
        eve2 = np.mean(np.array(eves2))
        eve2 = int(eve2)
        return eve1, eve2










    def compute_homography(self,src_points, dst_points):
        """
        计算两组四个对应点之间的单应性矩阵
        输入:
            src_points: 源点列表，格式为 [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
            dst_points: 目标点列表，格式同上
        输出:
            H: 3x3 单应性矩阵


            H13=50 图像向右平移50像素
            H23=-30 图像向下平移30像素
        """
        assert len(src_points) == 4 and len(dst_points) == 4, "需要四对对应点"

        # 构建系数矩阵 A 和常数项 b
        A = []
        b = []
        for (x, y), (xp, yp) in zip(src_points, dst_points):
            A.append([x, y, 1, 0, 0, 0, -x * xp, -y * xp])
            A.append([0, 0, 0, x, y, 1, -x * yp, -y * yp])
            b.extend([xp, yp])

        A = np.array(A, dtype=np.float64)
        b = np.array(b, dtype=np.float64)

        # 解线性方程组 Ah = b
        try:
            h = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # 若矩阵奇异，使用最小二乘法
            h, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        # 构造单应性矩阵 H
        H = np.array([
            [h[0], h[1], h[2]],
            [h[3], h[4], h[5]],
            [h[6], h[7], 1.0]
        ], dtype=np.float64)

        return H