""" Various helper functions. """
import cv2
import itertools
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from common import diag
from common import geometry
from common import utils as cu


class Axes:
    """ Axes to draw in make_log_image() """
    def __init__(self, rot, transl=None, length=100, width=1):
        self.rot = rot
        self.transl = transl
        self.length = length
        self.width = width


class Dir:
    """ Direction vector to draw in make_log_image()  """
    def __init__(self, dir_vec, transl=None, length=100, width=1, color=(1, 1, 0)):
        self.dir_vec = dir_vec
        self.transl = transl
        self.length = length
        self.width = width
        self.color = color


class Confidence:
    """ Confidence value to draw in make_log_image(), in red/blue colors for negative/positive values. """
    def __init__(self, confidence, offset=None, width=5):
        """
        Creates object.
        :param confidence: a vector (B, 1)  of confidence values in range [-1, 1] (or higher).
        """
        self.confidence = confidence
        self.offset = width // 2 if offset is None else offset
        self.width = width

@torch.no_grad()
def make_log_image(image, tag=None, *args):
    """
    Prepares an image for logging by drawing head poses.
    :param image: image tensor (B, 3, H, W) in range [-1, 1].
    :param tag: string or list of strings to show on the image or list of tags for every batch element.
    :param args: additional objects to draw.
    :return: an image as numpy array (B, 3, H, W)
    """
    b, c, h, w = image.shape
    # Draw axis for visualization only, preserve the original data.
    image = cu.range_2_1(image).moveaxis(1, -1).cpu().numpy()
    image = np.ascontiguousarray(image)

    def make_transl(transl):
        transl = torch.zeros(b, 2) if transl is None else transl
        return geometry.norm_to_pixel2(transl, (h, w)).detach().cpu().numpy()

    if c == 1:
        image = image.repeat(3, -1)
    for arg in args:
        if isinstance(arg, Axes):
            rot = geometry.h3(arg.rot)[:, :3, :3].detach().cpu().numpy()
            transl = make_transl(arg.transl)
            for i in range(len(image)):
                cu.draw_axes(image[i], rot[i], transl[i, 0], transl[i, 1],
                             colors=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                             length=arg.length, width=arg.width)

        elif isinstance(arg, Dir):
            dir_vec = arg.dir_vec[:, :2].detach().cpu().numpy()
            transl = make_transl(arg.transl)
            for i in range(len(image)):
                cu.draw_direction_vector(image[i], dir_vec[i], tx=transl[i, 0], ty=transl[i, 1],
                                         color=arg.color, length=arg.length, width=arg.width)
        elif isinstance(arg, Confidence):
            for i in range(len(image)):
                y0 = image[i].shape[0] // 2
                try:
                    confidence = arg.confidence[i].item()
                    y = int(y0 * (1 + confidence))
                except ValueError:
                    # In case of nan, etc.
                    y = 0
                color = (1, 0, 0) if confidence < 0 else (0, 0, 1)
                cv2.line(image[i], (arg.offset, y0), (arg.offset, y), color, arg.width)
        else:
            raise ValueError('Unexpected type')


    if tag is not None:
        if not isinstance(tag, (list, tuple)):
            tag = [tag] * len(image)
        for i in range(len(image)):
            text_params = {
                'fontFace': cv2.FONT_HERSHEY_SIMPLEX,
                'fontScale': 1.0,
                'thickness': 1,
                'color': (0, 255, 0)
            }
            cv2.putText(image[i], f'{tag[i]}', (2, 30), **text_params)

    image = np.moveaxis(image, -1, 1)
    image = np.clip(image, 0, 1)  # A model can occasionally generate out-of range colors.

    return image


@torch.no_grad()
def make_standard_image(input_image, name, size):
    """ Converts a visualization image to a labeled image of given size, with number of channnels 3.
        E.g. a flow image 8x8 will be resized to 256x256 and labeled.
    """
    result = input_image.clamp(0, 1)
    if result.shape[1] == 1:
        result = result.expand(-1, 3, -1, -1)
    result = cu.interpolate(result, size=size, mode='nearest').movedim(1, -1).cpu().numpy()
    result = np.ascontiguousarray(result)
    text_params = {
        'fontFace': cv2.FONT_HERSHEY_SIMPLEX,
        'fontScale': 0.75,
        'thickness': 1,
        'color': (0, 1, 0)
    }
    for i in range(len(result)):
        cv2.putText(result[i], name, (2, 30), **text_params)
    return result


@torch.no_grad()
def visualize_vectors(vectors, h, scale_h=4, scale_w=8, pad_w=1, scale_value=1):
    """ Visualize a list of vectors.
    :param vectors: a list or tuple of vectors (B, D), all B must be equal, D might be different.
    :param h: height of the result.
    :param scale_h: scale factor in height dimension.
    :param scale_w: scale factor in width dimension.
    :param ws: scale factor in width dimension.
    :param pad_w: pad in width dimension to separate vectors.
    :param scale_value: scale value.
    """
    if h % scale_h != 0:
        raise ValueError('h must be divisible by scale_h')
    hs = h // scale_h
    if not isinstance(vectors, (list, tuple)):
        vectors = (vectors,)
    vis = []
    for v in vectors:
        v = cu.visualize_sign((v * scale_value).unsqueeze(1))
        b, c, d = v.shape
        num_cols = d // hs
        if d % hs != 0:
            num_cols += 1
            v = F.pad(v, (0, hs - (d % hs)), value=1)
        v = v.reshape(b, c, num_cols, hs).transpose(2, 3)
        v = cu.interpolate(v, size=(h, num_cols * scale_w), mode='nearest')
        v = F.pad(v, (0, pad_w), value=1)
        vis.append(v)
    vis = torch.cat(vis, -1)
    vis = vis.movedim(1, -1).cpu().numpy()
    return vis


class StatsStorage:
    """ Collects tensors, e.g. to log histograms. """

    class Subgroup:
        def __init__(self, update_interval=1):
            self.values = []
            self.update_count = 0
            self.update_interval = update_interval

        def update(self, value):
            if self.update_count % self.update_interval == 0:
                self.values.append(value)

    def __init__(self, config):
        self._groups = {}
        self._config = config
        self._update_interval = config.get('update_interval', 1)
        # As the variance of a value can vary a lot over time, tensorboard histograms may become unreadable.
        # E.g. if a value varies in range [-100, 100] in the iterations 0-4 (see diag.iteration),
        # and in range [-0.1, 0.1] in the iterations 5-9, the 2nd step histograms will be very narrow.
        # This settings can group multiple steps together into  to make the histograms more readable.
        # For the example above it should be set to 5.
        self._tensorboard_hist_iteration_group = config.get('tensorboard_hist_iteration_group', None)

    @torch.no_grad()
    def update(self, group, subgroup, value):
        """
        Update data.
        :param group: group of data, e.g. head_pose. All tensors within a group
            shall have the same dimension except the batch one.
        :param subgroup: subgroup, e.g. we can collect the stats for head poses from multiple datasets.
        :param value: a tensor [B, D] or a list of such tensors.
        """
        if isinstance(value, (list, tuple)):
            value = torch.cat(value)
        value = value.detach().cpu()
        if value.ndim != 2:
            raise ValueError('Value must have 2 dimensions')
        if group not in self._groups:
            self._groups[group] = {}
        group_data = self._groups[group]
        if subgroup not in group_data:
            group_data[subgroup] = StatsStorage.Subgroup(self._update_interval)
        group_data[subgroup].update(value)

    def get_dim_name(self, group_name, dim):
        dim_names = self._config.get('groups', {}).get(group_name, {}).get('dim_names', None)
        if dim_names:
            return dim_names[dim]
        return str(dim)

    @torch.no_grad()
    def log(self, file=None):
        """ Combines all data, logs it and resets the storage for the next iteration. """

        # Concatenate all values and compute group totals
        for group_name, group in self._groups.items():
            group_total = []
            for subgroup_name, subgroup in group.items():
                subgroup.values = torch.cat(subgroup.values)
                group_total.append(subgroup.values)
            if len(group_total) > 1:
                group['total'] = StatsStorage.Subgroup(0)
                group['total'].values = torch.cat(group_total)

        # Log data
        torch.set_printoptions(linewidth=9999, threshold=99999, sci_mode=False)
        for group_name, group in self._groups.items():
            for subgroup_name, subgroup in group.items():
                for d in range(subgroup.values.shape[1]):
                    tb_it_group = self._tensorboard_hist_iteration_group
                    if tb_it_group:
                        group_text = f'[{diag.iteration // tb_it_group * tb_it_group}]'
                    else:
                        group_text = ''
                    diag.add_histogram(
                        f'{group_name}{group_text}/{subgroup_name}.{self.get_dim_name(group_name, d)}',
                        subgroup.values[:, d])

                print_title(f'{group_name}/{subgroup_name}, num: {len(subgroup.values)}', file=file)
                print('mean: ', subgroup.values.mean(0), file=file)
                cov = torch.cov(subgroup.values.T)
                if cov.ndim == 0:
                    cov = cov.reshape(1, 1)
                print('orig var:', cov.diag(), file=file)
                try:
                    v, _ = stats.pca(cov)
                    print('pca  var:', v, file=file)
                except Exception as e:
                    print('pca failed', file=file)
                    print(e, file=file)

                print('cov:', file=file)
                print(cov, file=file)
        torch.set_printoptions(profile='default')
        self._plot_covariance()

        # Reset for the next iteration.
        self._groups = {}

    def _plot_covariance(self):
        groups_cfg = self._config.get('groups', {})
        for group_name, group in self._groups.items():
            if group_name not in groups_cfg or not groups_cfg[group_name].get('cov_histogram', False):
                continue
            for subgroup_name, subgroup in group.items():
                dims = range(subgroup.values.shape[1])
                dim_names = groups_cfg[group_name].get('dim_names', list(str(x) for x in dims))
                for comb in itertools.combinations(dims, 2):
                    xval = subgroup.values[:, comb[0]].numpy()
                    yval = subgroup.values[:, comb[1]].numpy()
                    heatmap, xedges, yedges = np.histogram2d(xval, yval, bins=10)
                    matplotlib.use('Agg')
                    fig, ax = plt.subplots()
                    xname = dim_names[comb[0]]
                    yname = dim_names[comb[1]]
                    plt.xlabel(xname)
                    plt.ylabel(yname)
                    plt.xticks(xedges)
                    plt.yticks(yedges)

                    def tick_formatter(x, pos):
                        return f'{x:.2f}'.replace('-0', '-').lstrip('0')

                    ax.xaxis.set_major_formatter(tick_formatter)
                    ax.yaxis.set_major_formatter(tick_formatter)
                    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                    im = ax.imshow(heatmap,
                                   extent=extent,
                                   origin='lower',
                                   cmap='viridis', interpolation='nearest')
                    fig.colorbar(im, ax=ax)
                    diag.add_figure(f'{group_name}/{subgroup_name}.{xname}-{yname}', fig)


def print_title(text, file, size=80, char='-'):
    text += ' '
    text += char * (size - len(text))
    print(text, file=file)


