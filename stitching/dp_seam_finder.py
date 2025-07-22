import numpy as np
import cv2
class DpSeamFinder:
    def __init__(self, cost_type="color_grad", vertical=True):
        self.cost_type = cost_type  # "color", "color_grad" 等
        self.vertical = vertical  # 接缝方向（垂直或水平）
        self.weights = None  # 权重矩阵（可选）

    def find(self, images, masks, corners):
        # 初始化接缝掩码
        seam_masks = [np.zeros(mask.shape, dtype=np.uint8) for mask in masks]

        # 处理每个重叠区域
        for idx1, idx2 in self.find_overlapping_pairs(corners):
            # 提取重叠区域
            roi1, roi2 = self.get_overlap_roi(images[idx1], images[idx2], corners[idx1], corners[idx2])

            # 计算能量图
            energy = self.compute_energy(roi1, roi2)

            # 应用动态规划
            seam = self.find_seam_dp(energy)

            # 更新接缝掩码
            self.update_seam_mask(seam_masks[idx1], seam_masks[idx2], seam, corners[idx1], corners[idx2])

        return seam_masks

    def compute_energy(self, img1, img2):
        # 颜色差异
        color_diff = self.compute_color_difference(img1, img2)

        # 梯度差异（如果是color_grad模式）
        if self.cost_type == "color_grad":
            grad_diff = self.compute_gradient_difference(img1, img2)
            # 组合颜色和梯度差异
            energy = 0.5 * color_diff + 0.5 * grad_diff
        else:
            energy = color_diff

        # 应用权重（如果有）
        if self.weights is not None:
            energy *= self.weights

        return energy

    def compute_color_difference(self, img1, img2):
        # 多通道颜色差异计算
        if img1.ndim == 3:
            return np.mean(np.abs(img1 - img2), axis=2)
        else:
            return np.abs(img1 - img2)

    def compute_gradient_difference(self, img1, img2):
        # 计算图像梯度
        def compute_grad(img):
            dx = cv2.Scharr(img, cv2.CV_32F, 1, 0)
            dy = cv2.Scharr(img, cv2.CV_32F, 0, 1)
            return np.sqrt(dx ** 2 + dy ** 2)

        grad1 = compute_grad(img1)
        grad2 = compute_grad(img2)

        # 多通道梯度差异
        if img1.ndim == 3:
            return np.mean(np.abs(grad1 - grad2), axis=2)
        else:
            return np.abs(grad1 - grad2)

    def find_seam_dp(self, energy):
        # 根据接缝方向选择处理方式
        if self.vertical:
            return self.find_vertical_seam(energy)
        else:
            return self.find_horizontal_seam(energy)

    def find_vertical_seam(self, energy_map):
        rows, cols = energy_map.shape
        # 累积能量矩阵
        dp = np.zeros((rows, cols), dtype=np.float32)
        # 路径记录矩阵
        path = np.zeros((rows, cols), dtype=np.int32)

        # 初始化第一行
        dp[0, :] = energy_map[0, :]

        # 动态规划计算最小累积能量
        for i in range(1, rows):
            for j in range(cols):
                # 考虑左上、正上、右上三个方向
                min_prev = dp[i - 1, j]
                min_index = j

                if j > 0 and dp[i - 1, j - 1] < min_prev:
                    min_prev = dp[i - 1, j - 1]
                    min_index = j - 1

                if j < cols - 1 and dp[i - 1, j + 1] < min_prev:
                    min_prev = dp[i - 1, j + 1]
                    min_index = j + 1

                # 更新累积能量和路径
                dp[i, j] = energy_map[i, j] + min_prev
                path[i, j] = min_index

        # 回溯找到最优路径
        seam = []
        j = np.argmin(dp[-1, :])  # 从最后一行能量最小的点开始
        for i in range(rows - 1, -1, -1):
            seam.append((i, j))
            j = path[i, j]

        return seam[::-1]  # 反转顺序，从上到下

    def find_horizontal_seam(self, energy):
        # 转置矩阵后使用垂直接缝方法
        return self.find_vertical_seam(energy.T)

    ### 这个是直接计算的
    def compute_data_term(self, img1, img2, alpha=0.7, beta=0.3):
        # 颜色差异
        if img1.ndim == 3:  # RGB图像
            color_diff = np.mean(np.abs(img1 - img2), axis=2)
        else:  # 灰度图像
            color_diff = np.abs(img1 - img2)

        # 梯度差异
        def compute_gradient(img):
            dx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
            dy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
            return np.sqrt(dx ** 2 + dy ** 2)

        grad1 = compute_gradient(img1)
        grad2 = compute_gradient(img2)

        if img1.ndim == 3:
            grad_diff = np.mean(np.abs(grad1 - grad2), axis=2)
        else:
            grad_diff = np.abs(grad1 - grad2)

        # 归一化
        color_diff = (color_diff - np.min(color_diff)) / (np.max(color_diff) - np.min(color_diff) + 1e-8)
        grad_diff = (grad_diff - np.min(grad_diff)) / (np.max(grad_diff) - np.min(grad_diff) + 1e-8)

        return alpha * color_diff + beta * grad_diff