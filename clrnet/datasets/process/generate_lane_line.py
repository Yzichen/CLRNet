import math
import numpy as np
import cv2
import imgaug.augmenters as iaa
from imgaug.augmentables.lines import LineString, LineStringsOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from scipy.interpolate import InterpolatedUnivariateSpline
from clrnet.datasets.process.transforms import CLRTransforms

from ..registry import PROCESS


@PROCESS.register_module
class GenerateLaneLine(object):
    def __init__(self, transforms=None, cfg=None, training=True):
        self.transforms = transforms
        self.img_w, self.img_h = cfg.img_w, cfg.img_h
        self.num_points = cfg.num_points
        self.n_offsets = cfg.num_points
        self.n_strips = cfg.num_points - 1
        self.strip_size = self.img_h / self.n_strips
        self.max_lanes = cfg.max_lanes
        self.offsets_ys = np.arange(self.img_h, -1, -self.strip_size)
        self.training = training

        if transforms is None:
            transforms = CLRTransforms(self.img_h, self.img_w)

        if transforms is not None:
            img_transforms = []
            for aug in transforms:
                p = aug['p']
                if aug['name'] != 'OneOf':
                    img_transforms.append(
                        iaa.Sometimes(p=p,
                                      then_list=getattr(
                                          iaa,
                                          aug['name'])(**aug['parameters'])))
                else:
                    img_transforms.append(
                        iaa.Sometimes(
                            p=p,
                            then_list=iaa.OneOf([
                                getattr(iaa,
                                        aug_['name'])(**aug_['parameters'])
                                for aug_ in aug['transforms']
                            ])))
        else:
            img_transforms = []
        self.transform = iaa.Sequential(img_transforms)

    def lane_to_linestrings(self, lanes):
        lines = []
        for lane in lanes:
            lines.append(LineString(lane))

        return lines

    def sample_lane(self, points, sample_ys):
        # this function expects the points to be sorted
        points = np.array(points)
        if not np.all(points[1:, 1] < points[:-1, 1]):
            raise Exception('Annotaion points have to be sorted')
        x, y = points[:, 0], points[:, 1]

        # interpolate points inside domain
        assert len(points) > 1
        interp = InterpolatedUnivariateSpline(y[::-1],
                                              x[::-1],
                                              k=min(3,
                                                    len(points) - 1))
        domain_min_y = y.min()
        domain_max_y = y.max()
        sample_ys_inside_domain = sample_ys[(sample_ys >= domain_min_y)
                                            & (sample_ys <= domain_max_y)]
        assert len(sample_ys_inside_domain) > 0
        interp_xs = interp(sample_ys_inside_domain)

        # extrapolate lane to the bottom of the image with a straight line using the 2 points closest to the bottom
        two_closest_points = points[:2]
        extrap = np.polyfit(two_closest_points[:, 1],
                            two_closest_points[:, 0],
                            deg=1)
        extrap_ys = sample_ys[sample_ys > domain_max_y]
        extrap_xs = np.polyval(extrap, extrap_ys)
        all_xs = np.hstack((extrap_xs, interp_xs))

        # separate between inside and outside points
        inside_mask = (all_xs >= 0) & (all_xs < self.img_w)
        xs_inside_image = all_xs[inside_mask]
        xs_outside_image = all_xs[~inside_mask]

        return xs_outside_image, xs_inside_image

    def filter_lane(self, lane):
        assert lane[-1][1] <= lane[0][1]
        filtered_lane = []
        used = set()
        for p in lane:
            if p[1] not in used:
                filtered_lane.append(p)
                used.add(p[1])

        return filtered_lane

    def transform_annotation(self, anno, img_wh=None):
        img_w, img_h = self.img_w, self.img_h

        old_lanes = anno['lanes']   # old_lanes: [[(x_00,y0), (x_01,y1), ...], [(x_10,y0), (x_11,y1), ...], ...]

        # removing lanes with less than 2 points
        old_lanes = filter(lambda x: len(x) > 1, old_lanes)
        # sort lane points by Y (bottom to top of the image)    # 图像底部-->顶部
        old_lanes = [sorted(lane, key=lambda x: -x[1]) for lane in old_lanes]
        # remove points with same Y (keep first occurrence)
        old_lanes = [self.filter_lane(lane) for lane in old_lanes]
        # normalize the annotation coordinates
        old_lanes = [[[
            x * self.img_w / float(img_w), y * self.img_h / float(img_h)
        ] for x, y in lane] for lane in old_lanes]
        # create tranformed annotations     (max_lanes, 6+S)
        lanes = np.ones(
            (self.max_lanes, 2 + 1 + 1 + 2 + self.n_offsets), dtype=np.float32
        ) * -1e5  # 2 scores, 1 start_y, 1 start_x, 1 theta, 1 length, S coordinates
        lanes_endpoints = np.ones((self.max_lanes, 2))      # (max_lanes, 2)
        # lanes are invalid by default
        lanes[:, 0] = 1
        lanes[:, 1] = 0
        for lane_idx, lane in enumerate(old_lanes):
            if lane_idx >= self.max_lanes:
                break

            try:
                xs_outside_image, xs_inside_image = self.sample_lane(
                    lane, self.offsets_ys)
            except AssertionError:
                continue
            if len(xs_inside_image) <= 1:
                continue
            all_xs = np.hstack((xs_outside_image, xs_inside_image))
            lanes[lane_idx, 0] = 0
            lanes[lane_idx, 1] = 1
            lanes[lane_idx, 2] = len(xs_outside_image) / self.n_strips       # normalized start_y
            lanes[lane_idx, 3] = xs_inside_image[0]     # img_w 尺度下的 start_x

            thetas = []
            for i in range(1, len(xs_inside_image)):
                theta = math.atan(
                    i * self.strip_size /
                    (xs_inside_image[i] - xs_inside_image[0] + 1e-5)) / math.pi
                theta = theta if theta > 0 else 1 - abs(theta)
                thetas.append(theta)

            theta_far = sum(thetas) / len(thetas)

            # lanes[lane_idx,
            #       4] = (theta_closest + theta_far) / 2  # averaged angle
            lanes[lane_idx, 4] = theta_far
            lanes[lane_idx, 5] = len(xs_inside_image)   # length
            lanes[lane_idx, 6:6 + len(all_xs)] = all_xs     # img_w 尺度下的 xs

            lanes_endpoints[lane_idx, 0] = (len(all_xs) - 1) / self.n_strips
            lanes_endpoints[lane_idx, 1] = xs_inside_image[-1]

        new_anno = {
            'label': lanes,    # (max_lanes, 6+S)
            'old_anno': anno,
            'lane_endpoints': lanes_endpoints    # (max_lanes, 2)
        }
        return new_anno

    def linestrings_to_lanes(self, lines):
        lanes = []
        for line in lines:
            lanes.append(line.coords)

        return lanes

    def __call__(self, sample):
        img_org = sample['img']
        if sample.get('lanes', None) is not None:
            line_strings_org = self.lane_to_linestrings(sample['lanes'])
            line_strings_org = LineStringsOnImage(line_strings_org,
                                                  shape=img_org.shape)
        else:
            line_strings_org = None

        for i in range(30):
            if self.training:
                mask_org = SegmentationMapsOnImage(sample['mask'],
                                                   shape=img_org.shape)
                img, line_strings, seg = self.transform(
                    image=img_org.copy().astype(np.uint8),
                    line_strings=line_strings_org,
                    segmentation_maps=mask_org)
            else:
                img, line_strings = self.transform(
                    image=img_org.copy().astype(np.uint8),
                    line_strings=line_strings_org)

            # print(img.shape)
            # print("!!!")
            # for line in line_strings:
            #     coords = line.coords
            #     print(coords[(coords[:, 0] >= 800)])
            #     print(coords[coords[:, 1] >= 320])
            # print("!!!")
            line_strings.clip_out_of_image_()
            # print("###")
            # for line in line_strings:
            #     coords = line.coords
            #     print(coords[(coords[:, 0] >= 800)])
            #     print(coords[coords[:, 1] >= 320])
            # print("###")

            new_anno = {'lanes': self.linestrings_to_lanes(line_strings)}
            try:
                annos = self.transform_annotation(new_anno,
                                                  img_wh=(self.img_w,
                                                          self.img_h))
                label = annos['label']
                lane_endpoints = annos['lane_endpoints']
                break
            except:
                if (i + 1) == 30:
                    self.logger.critical(
                        'Transform annotation failed 30 times :(')
                    exit()

        # for vis
        # import cv2
        # img_copy = img.copy()
        # lanes = annos['label']
        # for lane in lanes:
        #     # lane: (bg, fg, start_y, start_x, theta, length, coordinates(S))
        #     if lane[1]:
        #         start_y = int(lane[2] * self.n_strips)
        #         length = lane[5]
        #         end_y = int(start_y + length - 1)
        #         assert end_y <= self.num_points
        #
        #         xs = lane[-self.num_points:]
        #         print(len(xs))
        #         valid_ys = self.offsets_ys[start_y:end_y]
        #         valid_xs = xs[start_y:end_y]
        #         for i in range(1, len(valid_ys)):
        #             print(i)
        #             print((round(valid_xs[i-1]), round(valid_ys[i-1])))
        #             print((round(valid_xs[i]), round(valid_ys[i])))
        #             cv2.line(img_copy, (round(valid_xs[i-1]), round(valid_ys[i-1])),
        #                      (round(valid_xs[i]), round(valid_ys[i])), (255, 0, 0), thickness=5)
        # cv2.imshow("line", img_copy)
        # cv2.waitKey(0)


        sample['img'] = img.astype(np.float32) / 255.
        sample['lane_line'] = label
        sample['lanes_endpoints'] = lane_endpoints
        sample['gt_points'] = new_anno['lanes']

        sample['seg'] = seg.get_arr() if self.training else np.zeros(
            img_org.shape)

        return sample
