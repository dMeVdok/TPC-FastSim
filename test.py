import uu
import matplotlib.pyplot as plt
from tensorflow.python.ops.gen_math_ops import Log1p
from models import scalers
import numpy as np
import tensorflow as tf
from models.nn import MomentsToImage, ImageToMoments
from data import preprocessing
from tqdm import tqdm

preprocessing._VERSION = 'data_v4'
data, features = preprocessing.read_csv_2d(pad_range=[-3,5], time_range=[-7,9])
features = features.astype('float32')

sc = scalers.Gaussian()

mti = MomentsToImage((8, 16))
itm = ImageToMoments()

mu0, mu1, s00, s01, s11, ig = [], [], [], [], [], []

mu0h, mu1h, s00h, s01h, s11h, igh = [], [], [], [], [], []

mu0d, mu1d, s00d, s11d, s01d, igd = [], [], [], [], [], []

marg_x_d, marg_y_d = [], []
marg_x_su, marg_y_su = [], []

inp = np.array([
    [0,0,-1,2,-1,10]
])


"""
fig, ax = plt.subplots(2,1,figsize=(12,8))
fig.suptitle("no activations, " + str(inp[0]))

ax[0].imshow(mti(tf.constant(inp, dtype=tf.float32))[0])
ax[0].set_xlabel("ITM: " + str(itm(mti(tf.constant(inp, dtype=tf.float32))).numpy()[0]))
ax[1].imshow(sc.unscale(inp, use_activations=True)[0])
ax[1].set_xlabel("Scale: " + str(sc.scale(sc.unscale(inp, use_activations=True))[0]))


"""

spoiled = np.zeros((8,16))

L1 = []

from matplotlib.patches import Rectangle

i = 0
for d in tqdm(data):
    i+=1
    img = np.array([d])
    inp = tf.constant(img, dtype=tf.float32)
    #img_reconstructed_layers = np.array(mti(itm(inp)))
    #img_reconstructed_scaler = sc.unscale(sc.scale(img))
    #rec_mom_layers = np.array(itm(inp))[0]
    rec_mom_scaler = sc.scale(img)[0]

    rec_mom_scale_unscale = sc.scale(sc.unscale(sc.scale(img)))[0]

    marg_x_d.append(np.sum(img[0], 0))
    marg_y_d.append(np.sum(img[0], 1))
    marg_x_su.append(np.sum(sc.unscale(sc.scale(img))[0], 0))
    marg_y_su.append(np.sum(sc.unscale(sc.scale(img))[0], 1))

    if (rec_mom_scaler[0] < 5.5) or (rec_mom_scaler[0] > 9.5) or (rec_mom_scaler[1] < 2.75) or (rec_mom_scaler[1] > 4.25):
        plt.imshow(img[0])
        plt.title(f"bad event #{i}\n{features[i]}")
        plt.colorbar()
        plt.gca().add_patch(Rectangle((5.5,2.75),9.5-5.5,4.25-2.75,linewidth=1,edgecolor='r',facecolor='none'))
        plt.savefig(f"bad_event_{i}.png")
        plt.clf()

    spoiled += (img - sc.unscale(sc.scale(img)))[0]

    mu0h.append(rec_mom_scaler[0])
    mu1h.append(rec_mom_scaler[1])
    s00h.append(rec_mom_scaler[2])
    s01h.append(rec_mom_scaler[3])
    s11h.append(rec_mom_scaler[4])
    igh.append(rec_mom_scaler[5])

    mu0d.append(rec_mom_scale_unscale[0] - rec_mom_scaler[0])
    mu1d.append(rec_mom_scale_unscale[1] - rec_mom_scaler[1])
    mu0.append(rec_mom_scaler[0])
    mu1.append(rec_mom_scaler[1]) 

    s00.append(rec_mom_scaler[2])
    s00d.append(rec_mom_scale_unscale[2] - rec_mom_scaler[2])
    s11.append(rec_mom_scaler[4])
    s11d.append(rec_mom_scale_unscale[4] - rec_mom_scaler[4])

    s01.append(rec_mom_scaler[3])
    s01d.append(rec_mom_scale_unscale[3] - rec_mom_scaler[3])
    ig.append(rec_mom_scaler[5])
    igd.append(rec_mom_scale_unscale[5] - rec_mom_scaler[5])
        #print(rec_mom_scale_unscale[5] - rec_mom_scaler[5])


    #mu0.append(rec_mom_layers[0]-rec_mom_scaler[0])
    #mu1.append(rec_mom_layers[1]-rec_mom_scaler[1])
    #s00.append(rec_mom_layers[2]-rec_mom_scaler[2])
    #s01.append(rec_mom_layers[3]-rec_mom_scaler[3])
    #s11.append(rec_mom_layers[4]-rec_mom_scaler[4])
    #ig.append(rec_mom_layers[5]-rec_mom_scaler[5])

print(np.min(mu0h), np.max(mu0h))
print(np.min(mu1h), np.max(mu1h))

def qv(x, y, u, v, bins=10):
    xn, yn, un, vn = [], [], [], []
    xa, ya, ua, va = np.array(x), np.array(y), np.array(u), np.array(v)
    l, r, t, b = xa.min(), xa.max(), ya.min(), ya.max()
    for xx in np.linspace(l, r, bins, endpoint=False):
        for yy in np.linspace(t, b, bins, endpoint=False):
            bw, bh = (r-l)/bins, (b-t)/bins
            uu = ua[(xa >= xx) & (xa < xx + bw) & (ya >= yy) & (ya < yy + bh)]
            vv = va[(xa >= xx) & (xa < xx + bw) & (ya >= yy) & (ya < yy + bh)]
            try:
                phi = np.arctan2(vv, uu)
                v, c = np.histogram(phi, bins=30)
                rho = np.mean(np.sqrt(uu**2 + vv**2))
                mode = c[np.argmax(v)]
                un.append(rho * np.cos(mode))
                vn.append(rho * np.sin(mode))
                xn.append(xx + bw / 2)
                yn.append(yy + bh / 2)
            except Exception:
                pass
    return xn, yn, un, vn

print(np.array(marg_x_d).shape)

plt.bar(list(range(len(marg_x_d[488]))), marg_x_d[488], fill=False, edgecolor="green", label="Data")
plt.bar(list(range(len(marg_x_su[488]))), marg_x_su[488], fill=False, edgecolor="red", label="Unscale(Scale(Data))")
plt.legend()

plt.savefig("marginal_x.png")

plt.clf()

plt.imshow(spoiled / len(data))
plt.title("signal pixel absolute discrepancy")
plt.colorbar()

plt.savefig("spoiled.png")
plt.clf()


plt.hist2d(mu0h, mu1h, bins=50)
plt.quiver(*qv(mu0, mu1, mu0d, mu1d, bins=25), color='r', angles='xy', scale_units='xy', scale=1, headaxislength=2, headlength=2)
plt.xlabel("mean x")
plt.ylabel("mean y")
plt.savefig("quiv_mu.pdf")
plt.clf()

plt.hist2d(s00h, s11h, cmin=1, bins=50)
plt.quiver(*qv(s00, s11, s00d, s11d, bins=25), color='r', angles='xy', scale_units='xy', scale=1)
plt.xlabel("Sq. Time Width")
plt.ylabel("Sq. Pad Width")
plt.ylim(0,0.6)
plt.savefig("quiv_sii.pdf")
plt.clf()

plt.hist2d(s01h, igh, bins=50)
plt.quiver(*qv(s01, ig, s01d, igd, bins=25), color='r', angles='xy', scale_units='xy', scale=1)
plt.xlabel("covariance")
plt.ylabel("log(integral)")
plt.savefig("quiv_sci.pdf")
plt.clf()

exit()

plt.xlabel("Mu0_layers(MTI(ITM(data))) - Mu0_scaler(unscale(scale(data)))")
plt.savefig("text_mu0.png")
plt.clf()

plt.hist(mu1, bins=100)
plt.xlabel("Mu1_layers(MTI(ITM(data))) - Mu1_scaler(unscale(scale(data)))")
plt.savefig("text_mu1.png")
plt.clf()

"""
plt.hist(s00, bins=100)
plt.xlabel("S00_layers(MTI(ITM(data))) - S00_scaler(unscale(scale(data)))")
plt.savefig("text_s00.png")
plt.clf()
plt.hist(s01, bins=100)
plt.xlabel("S01_layers(MTI(ITM(data))) - S01_scaler(unscale(scale(data)))")
plt.savefig("text_s01.png")
plt.clf()
plt.hist(s11, bins=100)
plt.xlabel("S11_layers(MTI(ITM(data))) - S11_scaler(unscale(scale(data)))")
plt.savefig("text_s11.png")
plt.clf()
plt.hist(ig, bins=100)
plt.xlabel("log(int)_layers(MTI(ITM(data))) - log(int)_scaler(unscale(scale(data)))")
plt.savefig("text_ig.png")
plt.clf()

plt.imshow(sc.unscale(sc.scale(img))[0])
plt.title("Unscale(Scale)")
plt.savefig("unscale-scale.png")

plt.imshow(np.array(mti(itm(inp)))[0])
plt.title("MTI(ITM)")
plt.savefig("mti-itm.png")
"""


#ax[0].imshow(mti(inp)[1].numpy())
#ax[0].set_xlabel("ImageToMoments->MomentsToImage")
#ax[1].imshow(sc.unscale(mmts, use_activations=True)[1])
#plt.imshow((mti(inp)[1].numpy() - sc.unscale(mmts, use_activations=True)[1]) / mti(inp)[1].numpy())
#plt.colorbar()


#print(np.sum(mti(inp)[1].numpy()), np.sum(sc.unscale(mmts, use_activations=True)[1]))
#print(np.isclose(mti(inp)[1].numpy(), sc.unscale(mmts, use_activations=True)[1]))

#ax[1].set_xlabel("Scale->Unscale")
#print(
#    np.sum(np.abs(sc.unscale(mmts, use_activations=True)[1], 
#    mti(inp)[1].numpy()))
#)
plt.savefig("./test.png")

# plot stats for all images from dataset 
# plot stats for moments
# check nans in plotting