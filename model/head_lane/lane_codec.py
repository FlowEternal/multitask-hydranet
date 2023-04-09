"""This script is used to process the auto lane dataset."""

import numpy as np
from scipy import interpolate
from head_lane.lane_spline_interp import spline_interp
from head_lane.lane_codec_utils import Point, Lane, get_lane_list, delete_repeat_y
from head_lane.lane_codec_utils import delete_nearby_point, trans_to_lane_with_type
from head_lane.lane_codec_utils import get_lane_loc_list, gettopk_idx

class LaneCodec(object):
    """This is the class of PointLaneCodec, which generate the groudtruth of every image pair.

    :param input_width: the width of input image
    :type input_width: float
    :param input_height: the height of input image
    :type input_height: float
    :param anchor_stride: the stride of every anchor
    :type anchor_stride: int
    :param points_per_line: the number of points in line
    :type points_per_line: int
    :param anchor_lane_num: how many lanes of every anchor
    :type anchor_lane_num: int
    """
    def __init__(self,
                 input_width,
                 input_height,
                 anchor_stride,
                 points_per_line,
                 do_interpolate=False,
                 anchor_lane_num=1,
                 scale_invariance=True,
                 ):
        self.input_width = input_width                              # 网络输入图像宽度
        self.input_height = input_height                            # 网络输入图像高度
        self.stride = anchor_stride                                 # cell相对于输入图像的边长
        self.feature_width = int(input_width / anchor_stride)       # 特征图宽度
        self.feature_height = int(input_height / anchor_stride)     # 特征图高度

        self.points_per_line = points_per_line                      # 一个anchor对应至多多少个水平线
        self.pt_nums_single_lane = 2 * points_per_line + 2          # 车道线回归头的维度

        self.points_per_anchor = points_per_line / self.feature_height
        self.interval = float(input_height) / points_per_line
        self.feature_size = self.feature_width * self.feature_height
        self.img_center_x = input_width / 2

        self.step_w = anchor_stride
        self.step_h = anchor_stride
        self.anchor_lane_num = anchor_lane_num      # 一个anchor预测至多多少个车道线
        self.interpolation = do_interpolate         # 是否将车道线延伸到板边
        self.scale_invariance = scale_invariance    # 是否使得预测结果尺度不变

    def encode_lane(self, lane_object, org_width, org_height):
        """Encode lane to target type.

        :param lane_object: lane annot
        :type lane_object: mon dict (a dict of special format)
        :param org_width: image width
        :type org_width: int
        :param org_height: image height
        :type org_height: int
        :return: gt_type: [576, class_num]
                 gt_loc:  [576, 146]
        :rtype: nd.array
        """
        s_x = self.input_width * 1.0 / org_width
        s_y = self.input_height * 1.0 / org_height
        gt_lanes_list = get_lane_list(lane_object, s_x, s_y)

        if len(gt_lanes_list) < 1:
            # background image
            gt_lane_offset = np.zeros(shape=(self.feature_size, self.points_per_line * 2 + 2), dtype=float)
            gt_lane_type = np.zeros(shape=(self.feature_size, 2), dtype=float)
            gt_lane_type[:, 0] = 1
            gt_loc = gt_lane_offset.astype(np.float32)
            gt_type = gt_lane_type.astype(np.float32)

        else:
            lane_set = trans_to_lane_with_type(gt_lanes_list)
            all_anchor_count = np.zeros(shape=(self.feature_height, self.feature_width))
            all_anchor_distance = list()
            all_anchor_loc = list()
            all_anchor_list = list()
            for idx,lane in enumerate(lane_set):
                cur_line = lane.lane
                new_lane = delete_repeat_y(cur_line)
                if len(new_lane) < 2:
                    startpos = -1
                    endpos = -1
                    x_list = []
                    y_list = []
                else:
                    interp_lane = spline_interp(lane=new_lane, step_t=1)
                    x_pt_list, y_pt_list = delete_nearby_point(interp_lane,self.input_width, self.input_height)
                    x_pt_list = x_pt_list[::-1]
                    y_pt_list = y_pt_list[::-1]  # y from small to big
                    startpos, endpos, x_list, y_list = self.uniform_sample_lane_y_axis(x_pt_list, y_pt_list)
                if startpos == -1 or endpos == -1:
                    continue
                anchor_list, anchor_distance_result, gt_loc_list = \
                    self.get_one_line_pass_anchors(startpos, endpos, x_list, y_list, all_anchor_count)

                all_anchor_distance.append(anchor_distance_result)
                all_anchor_loc.append(gt_loc_list)
                all_anchor_list.append(anchor_list)

            # process gt offset value
            if self.anchor_lane_num == 1:
                gt_type, gt_loc = self.get_one_lane_gt_loc_type(all_anchor_distance,all_anchor_loc, all_anchor_count)

            else:
                return None, None

        return gt_type, gt_loc

    def decode_lane(self,
                    predict_type,
                    predict_loc,
                    exist_threshold = 0.5,
                    margin_width=100.0,
                    ):
        """Decode lane to normal type.

        :param predict_type: class result of groundtruth
        :type predict_type: nd.array whose shape is [576, class_num]
        :param predict_loc: regression result of groundtruth
        :type predict_loc: nd.array whose shape is [576, 145]=[576, 72+1+72]
        :param exist_threshold: threshold to determine wheather there is a lane
        :type exist_threshold: float
        :param margin_width:
        :type margin_width: float
        :return: lane set
        :rtype: dict
        """
        predict_type = predict_type.detach().cpu().numpy()
        predict_loc = predict_loc.detach().cpu().numpy()

        lane_set = list()
        for h in range(self.feature_height):
            for w in range(self.feature_width):
                index = h * self.feature_width + w
                prob = predict_type[index][1]

                if prob < exist_threshold:
                    continue

                # process
                anchor_y_pos = int((self.feature_height - 1 - h) * self.points_per_anchor)
                anchor_center_x = (1.0 * w + 0.5) * self.step_w
                anchor_center_y = (1.0 * h + 0.5) * self.step_h

                #---------------------------------------------------#
                #  process up down anchor
                #---------------------------------------------------#
                up_lane = np.array([])
                down_lane = np.array([])

                end_pos = anchor_y_pos
                start_pos = anchor_y_pos

                up_anchor_lane = predict_loc[index, self.points_per_line + 2: 2 * self.points_per_line + 2]
                relative_end_pos_up = predict_loc[index, self.points_per_line + 1]  # up

                # up anchor
                if self.scale_invariance:
                    up_anchor_lane = predict_loc[index, self.points_per_line+2: 2 * self.points_per_line+ 2] * self.interval

                for i in range(self.points_per_line):
                    if i >= relative_end_pos_up or anchor_y_pos + i >= self.points_per_line:
                        break
                    rela_x = up_anchor_lane[i]
                    abs_x = anchor_center_x + rela_x

                    # TODO 这里保证x在范围以内
                    if abs_x < 0 or abs_x >= self.input_width:
                        break

                    abs_y = self.input_height - 1 - (anchor_y_pos + i) * self.interval
                    p = Point(abs_x, abs_y)
                    up_lane = np.append(up_lane, p)
                    end_pos = anchor_y_pos + i + 1 # end pose for up anchor

                # down anchor
                down_anchor_lane = predict_loc[index, :self.points_per_line]
                relative_end_pos_down = predict_loc[index, self.points_per_line]  # down

                if self.scale_invariance:
                    down_anchor_lane = predict_loc[index, :self.points_per_line] * self.interval

                for i in range(anchor_y_pos):
                    if i >= relative_end_pos_down or anchor_y_pos - 1 - i <0: # modify
                        break

                    rela_x = down_anchor_lane[i]
                    abs_x = anchor_center_x + rela_x

                    # TODO 这里保证x在范围以内 + margin保证车道线接触版边
                    if abs_x < 0 or abs_x >= self.input_width + margin_width:
                        break

                    abs_y = self.input_height - 1 - (anchor_y_pos - 1 - i) * self.interval
                    p = Point(abs_x, abs_y)
                    down_lane = np.append(p, down_lane)
                    start_pos = anchor_y_pos - 1 - i

                # merge anchor
                if up_lane.size + down_lane.size >= 2:
                    lane = np.append(down_lane, up_lane)
                    lane_predict = Lane(prob,
                                        start_pos,
                                        end_pos,
                                        anchor_center_x,
                                        anchor_center_y,
                                        1,
                                        lane)

                    lane_set.append(lane_predict)

        return lane_set

    def get_one_lane_gt_loc_type(self, all_anchor_distance, all_anchor_loc, all_anchor_count):
        gt_lane_offset = np.zeros(shape=(self.feature_size, self.pt_nums_single_lane), dtype=float)
        gt_lane_type = np.zeros(shape=(self.feature_size, 2), dtype=float)
        gt_lane_type[:, 0] = 1

        for h in range(self.feature_height):
            for w in range(self.feature_width):
                index = h * self.feature_width + w
                cnt = all_anchor_count[h][w]
                gt_loc_list, gt_dist_list = \
                    get_lane_loc_list(all_anchor_distance, all_anchor_loc, h, w)

                if cnt == 0:  # back ground
                    gt_lane_type[index, 0] = 1
                elif cnt == 1:  # single
                    gt_lane_type[index, 0] = 0
                    gt_lane_type[index, 1] = 1
                    gt_lane_offset[index, :self.pt_nums_single_lane] = gt_loc_list[0]
                else:  # choose one
                    gt_lane_type[index, 0] = 0
                    gt_lane_type[index, 1] = 1
                    # choose small distance
                    line_loc_num = len(gt_loc_list)
                    line_dist_num = len(gt_dist_list)
                    assert (line_dist_num == line_loc_num)
                    [top_idx] = gettopk_idx(gt_dist_list)
                    gt_lane_offset[index, :self.pt_nums_single_lane] = gt_loc_list[top_idx]

        gt_loc = gt_lane_offset.astype(np.float32)
        gt_type = gt_lane_type.astype(np.float32)

        return gt_type, gt_loc

    def uniform_sample_lane_y_axis(self, x_pt_list, y_pt_list):
        """Ensure y from bottom of image."""
        if len(x_pt_list) < 2 or len(y_pt_list) < 2:
            return -1, -1, [], []

        if self.interpolation:
            max_y = y_pt_list[-1]
            if max_y < self.input_height - 1:
                y1 = y_pt_list[-2]
                y2 = y_pt_list[-1]
                x1 = x_pt_list[-2]
                x2 = x_pt_list[-1]

                while max_y < self.input_height - 1:
                    y_new = max_y + self.interval
                    x_new = x1 + (x2 - x1) * (y_new - y1) / (y2 - y1)
                    x_pt_list.append(x_new)
                    y_pt_list.append(y_new)
                    max_y = y_new

        x_list = np.array(x_pt_list)
        y_list = np.array(y_pt_list)  # y from small to big
        if y_list.max() - y_list.min() < 5:  # filter < 5 pixel lane
            return -1, -1, [], []
        if len(y_list) < 4:
            tck = interpolate.splrep(y_list, x_list, k=1, s=0)
        else:
            tck = interpolate.splrep(y_list, x_list, k=3, s=0)

        if self.interpolation:
            startpos = 0
        else:
            startpos = int( (self.input_height - 1 -  y_list[-1]) / self.interval + 1)  # TODO 这里要加1

        endpos = int((self.input_height - 1 - y_list[0]) / self.interval)
        if endpos > self.points_per_line - 1:
            endpos = self.points_per_line - 1
        if startpos >= endpos:
            return -1, -1, [], []

        y_list = []
        expand_pos = endpos
        for i in range(startpos, expand_pos + 1):
            y_list.append(self.input_height - 1 - i * self.interval)
        xlist = interpolate.splev(y_list, tck, der=0)

        for i in range(len(xlist)):
            if xlist[i] == 0:
                xlist[i] += 0.01

        return startpos, endpos, xlist, y_list

    def get_one_line_pass_anchors(self, startpos, endpos, xlist, y_list, anchor_count):
        """Get one line pass all anchors."""
        anchor_list = []
        anchor_distance_result = []
        Gt_loc_list = []

        for i in range(0, endpos - startpos + 1):
            h = self.feature_height - 1 - int((startpos + i) * self.interval / self.step_h)
            w = int(xlist[i] / self.step_w)  # IndexError: list index out of range
            if h < 0 or h > self.feature_height - 1 or w < 0 or w > self.feature_width - 1:
                continue
            if (h, w) in anchor_list:
                continue
            anchor_y = (1.0 * h + 0.5) * self.step_h
            center_x = (1.0 * w + 0.5) * self.step_w

            # ensure anchor on same side of lane
            curr_y = self.input_height - 1 - (i + startpos) * self.interval
            if curr_y <= anchor_y:
                continue

            anchor_list.append((h, w))

            if self.interpolation:
                center_y = y_list[ int(self.points_per_line / self.feature_height)  * (self.feature_height - 1 - h)]
            else:
                # center_y = self.input_height - 1 - (i + startpos) * 4
                center_y = self.input_height -1 - (self.feature_height - 1 - h) * int(self.points_per_line / self.feature_height) * self.interval

            # get lane offset
            loss_line = [0] * (self.points_per_line * 2 + 2)
            length = endpos - startpos + 1
            # offset up cur anchor TODO
            up_index = 0
            for j in range(0, length):
                if y_list[j] <= center_y:
                    loss_line[self.points_per_line + 2 + up_index] = xlist[j] - center_x
                    up_index += 1
            loss_line[self.points_per_line + 1] = up_index
            # offset done cur anchor
            down_index = length - up_index - 1


            down_counter = 0
            for j in range(0, length):
                if y_list[ j] > center_y:
                    if xlist[j] - center_x == 0:
                        loss_line[down_index] = 0.000001
                    else:
                        loss_line[down_index] = xlist[j] - center_x
                    down_counter+=1
                    down_index -= 1
            # TODO modify zdx
            loss_line[self.points_per_line] = down_counter

            Gt_loc_list.append(loss_line)
            anchor_count[h][w] += 1
            distance = xlist[i] - self.img_center_x
            anchor_distance_result.append((h, w, distance))

        return anchor_list, anchor_distance_result, Gt_loc_list
