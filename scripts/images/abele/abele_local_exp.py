import numpy as np
import matplotlib.pyplot as plt
import time
from skimage.color import gray2rgb
from skimage import feature, transform

from ilore.ilorem import ILOREM
from ilore.util import neuclidean

from exputil import get_dataset
from exputil import get_black_box
from exputil import get_autoencoder

import warnings
warnings.filterwarnings('ignore')


def main():

    random_state = 0
    dataset = 'cifar10'
    black_box = 'RF'

    ae_name = 'aae'

    path = '../'
    path_models = path + 'models/'
    path_aemodels = path + 'aemodels/%s/%s/' % (dataset, ae_name)

    black_box_filename = path_models + '%s_%s' % (dataset, black_box)
    print(black_box_filename)
    _, _, X_test, Y_test, use_rgb = get_dataset(dataset)
    bb_predict, bb_predict_proba = get_black_box(black_box, black_box_filename, use_rgb)
    print(path_aemodels)
    print(dataset)
    ae = get_autoencoder(X_test, ae_name, dataset, path_aemodels)
    ae.load_model()

    class_name = 'class'
    class_values = ['%s' % i for i in range(len(np.unique(Y_test)))]

    i2e = 24
    img = X_test[i2e]
    print(img.shape)
    start = time.time()
    explainer = ILOREM(bb_predict, class_name, class_values, neigh_type='hrg', use_prob=True, size=300, ocr=0.1,
                       kernel_width=None, kernel=None, autoencoder=ae, use_rgb=use_rgb, valid_thr=0.5,
                       filter_crules=False, random_state=random_state, verbose=True, alpha1=0.5, alpha2=0.5,
                       metric=neuclidean, ngen=100, mutpb=0.2, cxpb=0.5, tournsize=3, halloffame_ratio=0.1,
                       bb_predict_proba=bb_predict_proba)

    exp = explainer.explain_instance(img, num_samples=200, use_weights=True, metric=neuclidean)
    end = time.time()
    print('print ', end - start)
    print('e = {\n\tr = %s\n\tc = %s    \n}' % (exp.rstr(), exp.cstr()))
    print(exp.bb_pred, exp.dt_pred, exp.fidelity)
    print(exp.limg)

    img2show, mask = exp.get_image_rule(features=None, samples=400)
    if use_rgb:
        plt.imshow(img2show, cmap='gray')
    else:
        plt.imshow(img2show)
    bbo = bb_predict(np.array([img2show]))[0]
    plt.title('Image to explain - black box %s' % bbo)
    #plt.show()
    plt.savefig(str(i2e)+'_to_explain'+str(dataset)+'.png')


    dx, dy = 0.05, 0.05
    xx = np.arange(0.0, img2show.shape[1], dx)
    yy = np.arange(0.0, img2show.shape[0], dy)
    xmin, xmax, ymin, ymax = np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)
    extent = xmin, xmax, ymin, ymax
    cmap_xi = plt.get_cmap('Greys_r')
    cmap_xi.set_bad(alpha=0)

    # Compute edges (to overlay to heatmaps later)
    percentile = 100
    dilation = 3.0
    alpha = 0.8
    xi_greyscale = img2show if len(img2show.shape) == 2 else np.mean(img2show, axis=-1)
    in_image_upscaled = transform.rescale(xi_greyscale, dilation, mode='constant')
    edges = feature.canny(in_image_upscaled).astype(float)
    edges[edges < 0.5] = np.nan
    edges[:5, :] = np.nan
    edges[-5:, :] = np.nan
    edges[:, :5] = np.nan
    edges[:, -5:] = np.nan
    overlay = edges

    print(mask.shape)
    print(overlay.shape)
    plt.imshow(mask, extent=extent, cmap=plt.cm.BrBG, alpha=1, vmin=0, vmax=255)
    plt.imshow(overlay, extent=extent, interpolation='none', cmap=cmap_xi, alpha=alpha)
    plt.axis('off')
    plt.title('Attention area respecting latent rule')
    #plt.show()
    plt.savefig(str(i2e)+'_explain_'+str(dataset)+'.png')



if __name__ == "__main__":
    main()
