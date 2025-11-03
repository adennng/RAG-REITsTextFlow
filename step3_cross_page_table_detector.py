# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os

class CrossPageTableDetector:
    def __init__(self):
        # 获取脚本所在目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # ====================== 
        # 可调节参数配置
        # ======================
        self.params = {
            # 文件处理参数
            'debug_dir': os.path.join(script_dir, "debug_output"),  # 调试输出目录改为脚本同目录下
            
            # 页脚/页眉剪裁参数（像素）
            'footer_crop': 250,    # 剪裁页面底部高度（页脚）
            'header_crop': 250,    # 剪裁页面顶部高度（页眉）
            
            # 检测区域参数（像素）
            'prev_bottom_region': 500,  # 前一页底部检测区域高度
            'next_top_region': 500,     # 下一页顶部检测区域高度
            
            # 图像预处理参数
            'gaussian_kernel': (5, 5),  # 高斯模糊核
            'block_size': 31,           # 自适应二值化块大小（必须为奇数）
            'c_param': 5,               # 自适应二值化C值
            
            # 线条检测参数
            'hough_threshold': 200,     # 霍夫变换阈值（值越大检测越严格）
            'vertical_min_length': 200,  # 竖线最小长度（像素）
            'horizontal_min_length': 200, # 横线最小长度（像素）
            'horizontal_edge_margin': 2000,      # 横线距离页面两侧的最小边距（过滤短横线）
            'horizontal_max_y_variance': 30,   # 横线Y坐标最大波动范围（过滤文字行）
            'max_line_gap': 5,         # 线段最大间隔（像素）
            'horizontal_angle_tolerance': 5,  # 横线允许的角度偏差（±15度）
            'vertical_angle_tolerance': 5,     # 竖线允许的角度偏差（±5度）
            'vertical_edge_margin': 50,  # 竖线距离页面左右边缘的最小距离（像素）
            
            # 跨页判断阈值
            'vertical_tolerance': 80,    # 竖线对齐容忍偏差（像素）
            'horizontal_margin': 300,     # 横线页边距阈值（像素）
            'min_vertical_lines': 3,     # 最小触发竖线数量
            'debug': True                # 调试模式
        }

        # 初始化调试目录
        if not os.path.exists(self.params['debug_dir']):
            os.makedirs(self.params['debug_dir'])

    # --------------------------
    # 基础方法（保持不变）
    # --------------------------
    def load_image(self, path):
        """支持中文路径的图像加载"""
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"图像加载失败: {path}")
        return img

    def preprocess_image(self, img):
        """图像预处理流水线（优化对比度和降噪）"""
        blurred = cv2.GaussianBlur(img, self.params['gaussian_kernel'], 0)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, self.params['block_size'], self.params['c_param']
        )

    def _debug_save_image(self, img, name):
        """保存调试图像（当前已注释，后续需要时取消注释）"""
        #if not self.params['debug']:
            #return
        #debug_path = os.path.join(self.params['debug_dir'], f"{name}.jpg")
        #cv2.imencode('.jpg', img)[1].tofile(debug_path)
        #print(f"[DEBUG] 保存图像: {debug_path} (尺寸: {img.shape[1]}x{img.shape[0]})")
        pass

    # --------------------------
    # 新增/修改的核心逻辑
    # --------------------------
    def _crop_regions(self, prev_img, next_img):
        """裁剪处理流程（增加详细日志）"""
        print("\n" + "="*40 + "\n[裁剪阶段]")
        
        # 前一页处理
        prev_region, prev_main = self._crop_and_validate(
            prev_img, self.params['footer_crop'], 
            self.params['prev_bottom_region'], 'footer'
        )
        print(f"前页裁剪：原始尺寸 {prev_img.shape[1]}x{prev_img.shape[0]} => "
              f"主裁剪后 {prev_main.shape[1]}x{prev_main.shape[0]} => "
              f"检测区域 {prev_region.shape[1]}x{prev_region.shape[0]}")

        # 下一页处理
        next_region, next_main = self._crop_and_validate(
            next_img, self.params['header_crop'], 
            self.params['next_top_region'], 'header'
        )
        print(f"后页裁剪：原始尺寸 {next_img.shape[1]}x{next_img.shape[0]} => "
              f"主裁剪后 {next_main.shape[1]}x{next_main.shape[0]} => "
              f"检测区域 {next_region.shape[1]}x{next_region.shape[0]}")

        return prev_region, next_region

    def _crop_and_validate(self, img, main_crop, region_height, region_type):
        """裁剪验证（增加尺寸校验）"""
        h, w = img.shape[:2]
        
        # 主裁剪
        if region_type == 'footer':
            main_cropped = img[:h - main_crop, :] if h > main_crop else img
        else:
            main_cropped = img[main_crop:, :] if h > main_crop else img
        
        # 检测区域裁剪
        h_main = main_cropped.shape[0]
        if region_type == 'footer':
            y_start = max(0, h_main - region_height)
            final_region = main_cropped[y_start:, :]
        else:
            y_end = min(region_height, h_main)
            final_region = main_cropped[:y_end, :]
        
        # 注释掉调试图像保存
        # self._debug_save_image(final_region, f"final_region_{region_type}")
        return final_region, main_cropped

    def _detect_horizontal_lines(self, img):
        """严格过滤文字行误判的横线检测"""
        processed = self.preprocess_image(img)
        edges = cv2.Canny(processed, 50, 150)
        
        # 霍夫变换检测线段
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180,
            threshold=self.params['hough_threshold'],
            minLineLength=self.params['horizontal_min_length'],
            maxLineGap=self.params['max_line_gap']
        )
        
        valid_lines = []
        debug_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # 条件1：线段长度要求
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if length < self.params['horizontal_min_length']:
                    continue
                
                # 条件2：线段需靠近两侧边缘（过滤文字行）
                if (x1 > self.params['horizontal_edge_margin'] and 
                    x2 < w - self.params['horizontal_edge_margin']):
                    continue
                
                # 条件3：线段Y坐标波动范围（过滤文字行）
                if abs(y1 - y2) > self.params['horizontal_max_y_variance']:
                    continue
                
                # 记录有效横线
                cv2.line(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                valid_lines.append((y1 + y2) // 2)  # 取平均Y坐标
                print(f"有效横线：长度={length:.1f}px, 位置Y={y1}-{y2}, 两端边距=({x1}, {w - x2})px")

        # 注释掉调试图像保存
        self._debug_save_image(debug_img, "detected_horizontal_lines")
        return valid_lines

    def _detect_vertical_lines(self, img, region_type):
        """检测竖线（新增边缘过滤条件）"""
        processed = self.preprocess_image(img)
        edges = cv2.Canny(processed, 50, 150)
        
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180,
            threshold=self.params['hough_threshold'],
            minLineLength=self.params['vertical_min_length'],
            maxLineGap=self.params['max_line_gap']
        )
        
        valid_lines = []
        debug_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_w = img.shape[1]  # 获取图像宽度
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                
                # 计算竖线平均X坐标
                x_avg = np.mean([x1, x2])
                
                # --------------------------
                # 过滤条件（新增边缘检测）
                # --------------------------
                # 条件1：角度符合要求
                angle_ok = 90 - self.params['vertical_angle_tolerance'] <= abs(angle) <= 90 + self.params['vertical_angle_tolerance']
                # 条件2：长度符合要求
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                length_ok = length >= self.params['vertical_min_length']
                # 条件3：不在边缘区域
                margin_ok = (x_avg > self.params['vertical_edge_margin']) and \
                            (x_avg < img_w - self.params['vertical_edge_margin'])
                
                if angle_ok and length_ok and margin_ok:
                    cv2.line(debug_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    valid_lines.append((x1, x2))
                    print(f"有效竖线：X={x_avg:.1f}, 角度={angle:.1f}°, 边缘距离={min(x_avg, img_w - x_avg):.1f}px")
                else:
                    # 打印过滤原因
                    reject_reasons = []
                    if not angle_ok: reject_reasons.append("角度不符")
                    if not length_ok: reject_reasons.append(f"长度不足({length:.1f}px)")
                    if not margin_ok: reject_reasons.append("靠近边缘")
                    print(f"过滤竖线：X={x_avg:.1f}, 原因={','.join(reject_reasons)}")
        
        # 注释掉调试图像保存
        # self._debug_save_image(debug_img, f"detected_vertical_{region_type}")
        return [np.mean([x1, x2]) for x1, x2 in valid_lines]

    def check_cross_page(self, prev_path, next_path):
        """跨页检测主逻辑（返回结构化结果）"""
        result = {
            'is_cross_page': False,
            'reason': None,
            'horizontal_distance': None,  # 保持为Python原生类型
            'vertical_avg_diff': None      # 保持为Python原生类型
        }


        try:
            prev_img = self.load_image(prev_path)
            next_img = self.load_image(next_path)
            print(f"图像加载成功：前页 {prev_path}\n{' '*20}后页 {next_path}")
        except Exception as e:
            print(f"错误: {e}")
            return result

        # 裁剪处理
        prev_region, next_region = self._crop_regions(prev_img, next_img)

        # ======================
        # 阶段1：检测下一页横线
        # ======================
        print("\n" + "="*40 + "\n[阶段1] 检测下一页横线")
        horizontal_lines = self._detect_horizontal_lines(next_region)
        is_horizontal_match = False
        
        if horizontal_lines:
            min_distance = min(horizontal_lines)
            print(f"最近横线距离顶部：{min_distance}px（阈值：{self.params['horizontal_margin']}px）")
            if min_distance <= self.params['horizontal_margin']:
                is_horizontal_match = True
                result['is_cross_page'] = True
                result['reason'] = 'horizontal'
                result['horizontal_distance'] = int(min_distance + self.params['header_crop'])  # 转换为Python int
                print("✅ 满足横线条件")

                

        # ======================
        # 阶段2：检测竖线对齐
        # ======================
        is_vertical_match = False
        vertical_log = []
        if not is_horizontal_match:
            print("\n" + "="*40 + "\n[阶段2] 检测竖线对齐")
            
            # 检测前页竖线
            prev_x = self._detect_vertical_lines(prev_region, "前页底部")
            # 检测后页竖线
            next_x = self._detect_vertical_lines(next_region, "后页顶部")
            
            if len(prev_x) >= self.params['min_vertical_lines'] and \
               len(next_x) >= self.params['min_vertical_lines']:
                diffs = []
                for p_x in prev_x:
                    closest = min(next_x, key=lambda x: abs(x - p_x))
                    diffs.append(abs(closest - p_x))
                avg_diff = np.mean(diffs)
                print(f"竖线平均偏差：{avg_diff:.1f}px（容忍度：{self.params['vertical_tolerance']}px）")
                
                if avg_diff <= self.params['vertical_tolerance']:
                    is_vertical_match = True
                    result['is_cross_page'] = True
                    result['reason'] = 'vertical'
                    result['vertical_avg_diff'] = float(avg_diff)  # 转换为Python float
                    print("✅ 满足竖线对齐条件")

        # ======================
        # 最终判断
        # ======================
        print("\n" + "="*40 + "\n[最终判断]")
        if result['is_cross_page']:
            print(f"判定结果：存在跨页表格（依据：{result['reason']}）")
        else:
            print("判定结果：不存在跨页表格")
        print("="*40)

        return result
    
            # 确保所有数值类型可序列化
        return {
            'is_cross_page': bool(result['is_cross_page']),
            'reason': str(result['reason']) if result['reason'] else None,
            'horizontal_distance': int(result['horizontal_distance']) if result['horizontal_distance'] is not None else None,
            'vertical_avg_diff': float(result['vertical_avg_diff']) if result['vertical_avg_diff'] is not None else None
        }

# ======================
# 使用示例
# ======================
if __name__ == "__main__":
    detector = CrossPageTableDetector()
    
    # 执行检测
    result = detector.check_cross_page(
        "***/508084.SH_汇添富九州通医药REIT\\2025-02-18_508084.SH_汇添富九州通医药REIT_汇添富九州通医药仓储物流封闭式基础设施证券投资基金基金合同生效公告\\table_image\\page_2.png",
        "***/508084.SH_汇添富九州通医药REIT\\2025-02-18_508084.SH_汇添富九州通医药REIT_汇添富九州通医药仓储物流封闭式基础设施证券投资基金基金合同生效公告\\table_image\\page_3-4.png"
    )
    print(f"\n检测结果：{result}")
    print(type(result['horizontal_distance']))  # 应显示<class 'int'>
    print(type(result['vertical_avg_diff']))    # 应显示<class 'float'>
