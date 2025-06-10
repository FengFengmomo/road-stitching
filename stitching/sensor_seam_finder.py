import cv2
import numpy as np


class SensorSeamFinder:
    def __init__(self):
        pass
    def find(self, imgs, corners, masks, sensor_masks = None):
        if sensor_masks is None:
            return
        imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in imgs] #转成灰度处理
        self.ranges = self.get_image_ranges(imgs, corners) # 获取每个图像的全局坐标范围

        # 获取重叠区域
        for i in range(len(self.ranges)-1):
            # 这里先计算0和1之间的
            overlap = self.get_overlap(self.ranges[i], self.ranges[i+1]) # 两个相邻图片之间应该有重叠区域
            if overlap is None:
                raise Exception("No overlap between images")
            # 获取重叠区域的像素点
            img1_roi, img2_roi = self.get_overlap_regions_pixels(imgs[i], corners[i], imgs[i+1], corners[i+1], overlap)
            seam_mask = self.calculate_seam_dp_horizontal(img1_roi, img2_roi)
            mask_above, mask_below =self.generate_seam_masks_horizontal(seam_mask)
            # 这里的上下位置在其他情况可能需要进行判断，但是在这里，第一张图片肯定是在下面
            self.update_overlap_masks(masks[i], corners[i], masks[i+1], corners[i+1], overlap, mask_below, mask_above)

        return masks

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

    # --------------------------------------------
    # 步骤2：计算两张图片的重叠区域
    # --------------------------------------------
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




    def calculate_seam_dp(self, img1_overlap, img2_overlap):
        """
        输入: img1_overlap 和 img2_overlap 是两张图片的重叠区域（尺寸相同）
        输出: 最佳接缝的掩码（True表示保留img1，False表示保留img2）
        """
        # 转换为浮点型以提高计算精度
        img1 = img1_overlap.astype(np.float32)
        img2 = img2_overlap.astype(np.float32)

        # 计算像素差异矩阵（使用颜色差异的平方和）
        diff = np.sum((img1 - img2) ** 2, axis=2)

        # 初始化DP表
        h, w = diff.shape # 计算出来色差的二维矩阵表
        dp = np.zeros_like(diff)
        dp[0, :] = diff[0, :]  # 第一行直接使用初始差异

        # 记录路径来源（用于回溯）
        path = np.zeros((h, w), dtype=np.int)

        # 填充DP表
        for i in range(1, h):
            for j in range(w):
                # 允许向左、中、右三个方向延伸（防止越界）
                left = max(0, j - 1)
                right = min(w - 1, j + 1)
                min_prev = np.min(dp[i - 1, left:right + 1]) # 从上一行的左右中选择最小的那个值
                prev_idx = np.argmin(dp[i - 1, left:right + 1]) + left # 返回这三个值里面的最小值的索引

                dp[i, j] = diff[i, j] + min_prev
                path[i, j] = prev_idx

        # 回溯路径
        seam_mask = np.ones((h, w), dtype=bool)  # 初始化为保留img1
        j = np.argmin(dp[-1, :])  # 最后一行最小差异位置

        for i in range(h - 1, -1, -1):
            seam_mask[i, j] = False  # 标记为使用img2的像素
            if i > 0:
                j = path[i, j]  # 更新上一行的列索引

        return seam_mask # 知道拼接缝，形成两张图片的掩码

    def calculate_seam_dp_horizontal(self, img1_overlap, img2_overlap):
        """
        输入: img1_overlap 和 img2_overlap 是两张图片的重叠区域（尺寸相同）
        输出: 最佳接缝的掩码（True表示保留img1，False表示保留img2）
        """
        # 转换为浮点型以提高计算精度
        img1 = img1_overlap.astype(np.float32)
        img2 = img2_overlap.astype(np.float32)

        # 计算像素差异矩阵（使用颜色差异的平方和）
        diff = np.sum((img1 - img2) ** 2, axis=2)

        # 初始化DP表（从左到右）
        h, w = diff.shape
        dp = np.zeros_like(diff)
        dp[:, 0] = diff[:, 0]  # 第一列直接使用初始差异

        # 记录路径来源（用于回溯）
        path = np.zeros((h, w), dtype=np.int)

        # 填充DP表（按列遍历）
        for j in range(1, w):
            for i in range(h):
                # 允许向上、中、下三个方向延伸（防止越界）
                top = max(0, i - 1)
                bottom = min(h - 1, i + 1)
                min_prev = np.min(dp[top:bottom + 1, j - 1])
                prev_idx = np.argmin(dp[top:bottom + 1, j - 1]) + top

                dp[i, j] = diff[i, j] + min_prev
                path[i, j] = prev_idx

        # 回溯路径（从右到左）
        seam_mask = np.ones((h, w), dtype=bool)  # 初始化为保留img1
        i = np.argmin(dp[:, -1])  # 最后一列最小差异的行索引

        for j in range(w - 1, -1, -1):
            seam_mask[i, j] = False  # 标记为使用img2的像素
            if j > 0:
                i = path[i, j]  # 更新前一列的行索引

        return seam_mask

    def generate_seam_masks_horizontal(self, seam_mask):
        """
        根据接缝掩码生成两个区域掩码：针对的是拼接缝是横着的
        - mask_above: 接缝线上方及接缝线为255
        - mask_below: 接缝线下方及接缝线为255
        输入:
            seam_mask: 布尔矩阵，True表示接缝线位置（形状为 HxW）
        输出:
            mask_above, mask_below: uint8类型掩码（0或255）
        """
        h, w = seam_mask.shape
        mask_above = np.zeros_like(seam_mask, dtype=np.uint8)
        mask_below = np.zeros_like(seam_mask, dtype=np.uint8)

        # 遍历每一行，找到接缝位置
        for row in range(h):
            # 找到当前行接缝线的列索引（可能有多个）
            seam_cols = np.where(seam_mask[row])[0]
            if len(seam_cols) == 0:
                # 若该行无接缝，则默认分割线为最右侧（根据需求调整）
                left_end = w
            else:
                # 取最左侧的接缝位置（或根据需求调整逻辑）
                left_end = seam_cols[0]

            # 填充上方掩码（左侧及接缝线）
            mask_above[row, :left_end + 1] = 255

            # 填充下方掩码（右侧及接缝线）
            mask_below[row, left_end:] = 255

        # 确保接缝线本身在两个掩码中均为255
        mask_above[seam_mask] = 255
        mask_below[seam_mask] = 255

        return mask_above, mask_below

    def generate_seam_masks_vetical(self, seam_mask):
        """
        根据接缝掩码生成两个区域掩码：针对的是拼接缝是竖着的
        - mask_above: 接缝线上方及接缝线为255
        - mask_below: 接缝线下方及接缝线为255
        输入:
            seam_mask: 布尔矩阵，True表示接缝线位置（形状为 HxW）
        输出:
            mask_above, mask_below: uint8类型掩码（0或255）
        """
        h, w = seam_mask.shape
        mask_left = np.zeros_like(seam_mask, dtype=np.uint8)
        mask_right = np.zeros_like(seam_mask, dtype=np.uint8)

        for col in range(w):
            seam_rows = np.where(seam_mask[:, col])[0]
            if len(seam_rows) == 0:
                top_end = h
            else:
                top_end = seam_rows[0]
            mask_left[:, col, :top_end + 1] = 255  # 上方区域
            mask_right[:, col, top_end:] = 255  # 下方区域

        # 确保接缝线本身在两个掩码中均为255
        mask_left[seam_mask] = 255
        mask_right[seam_mask] = 255

        return mask_left, mask_right

    def update_overlap_masks(self, mask1, corner1, mask2, corner2, overlap_global, seam_mask1, seam_mask2):
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
        x1_start, y1_start, x1_end, y1_end = clamp_coordinates(mask1, x1_min, y1_min, x1_max, y1_max)
        # 获取img2的有效区域
        x2_start, y2_start, x2_end, y2_end = clamp_coordinates(mask2, x2_min, y2_min, x2_max, y2_max)

        mask1[y1_start:y1_end, x1_start:x1_end] = seam_mask1

        mask2[y2_start:y2_end, x2_start:x2_end] = seam_mask2

if __name__ == '__main__':

    # 输入数据
    corners = [(-189, -67), (-193, -128), (-199, -165), (-205, -199)]
    image_width = 261
    image_height = 386
    imgs = [np.zeros((image_height, image_width), dtype=np.uint8) for _ in range(len(corners))]

    finder =  SensorSeamFinder()
    finder.find(imgs, corners, None)
    finder.get_overlap(finder.ranges[0], finder.ranges[1])
    # 打印结果
    print("各图片坐标范围:")
    for i, (pt1, pt2) in enumerate(finder.ranges):
        print(f"图片{i+1}: ({pt1[0]}, {pt1[1]}) -> ({pt2[0]}, {pt2[1]})")

    # --------------------------------------------
    # 示例使用：计算图片1和图片2的重叠区域
    # --------------------------------------------
    # 输入需要比较的两个图片索引（从0开始）
    idx1 = 0  # 图片1的索引
    idx2 = 1  # 图片2的索引

    if idx1 < 0 or idx1 >= len(finder.ranges) or idx2 < 0 or idx2 >= len(finder.ranges):
        print("错误：输入的索引超出范围")
    else:
        overlap = finder.get_overlap(finder.ranges[idx1], finder.ranges[idx2])
        if overlap is None:
            print(f"图片{idx1+1}与图片{idx2+1}无重叠区域")
        else:
            print(f"\n图片{idx1+1}与图片{idx2+1}的重叠区域:")
            print(f"X轴范围: {overlap[0]} ~ {overlap[2]} (宽度: {overlap[2] - overlap[0]})")
            print(f"Y轴范围: {overlap[1]} ~ {overlap[3]} (高度: {overlap[3] - overlap[1]})")


    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots()
    colors = ['red', 'blue', 'green', 'yellow']

    # 绘制所有图片范围
    for i, ((x1, y1), (x2, y2)) in enumerate(finder.ranges):
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                linewidth=1, edgecolor=colors[i],
                                facecolor=colors[i],
                                label=f'Image {i+1}')
        ax.add_patch(rect)

    # 设置坐标轴
    ax.set_xlim(-250, 100)
    ax.set_ylim(-250, 350)
    ax.legend()
    plt.gca().invert_yaxis()  # 图像坐标系Y轴向下
    plt.show()

    global_point = (0, 0)
    pixel_map = finder.get_corresponding_pixels(global_point, corners, imgs)
    for img_idx, (local_pt, pixel) in pixel_map.items():
        print(f"图片{img_idx + 1}: 局部坐标={local_pt}, 像素值={pixel}")
