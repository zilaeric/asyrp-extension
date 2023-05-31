from torchvision.utils import make_grid
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# constants
FIGURESPATH = "../../figures"
RUNSPATH = "../../src/runs"


#########
# UTILS #
#########

def gallery(array, ncols=3):
    nindex, height, width, intensity = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols, intensity))
    return result

def deconstruct_grid(img_grid):
    img_sz = img_grid.shape[1]-2
    img_num = int((img_grid.shape[0]-1) / (img_sz+1))
    
    imgs = []
    for i in range(img_num):
        vfrom = i*(img_sz+1) + 1
        vto = (i+1)*(img_sz+1)

        imgs.append(img_grid[vfrom:vto, 1:img_sz, :])

    return imgs


################
# REPRODUCTION #
################

def reproduction_in(img_ids, man_woman, dt_lambda=0.5, whitespace=30):
    """Function to reproduce samples using in-domain attributes.

    Args:
        img_ids (list): List of image IDs.
        man_woman (list): List of strings identifying required man/woman change.
    """
    attrs = {
        "Real image": "original", 
        "Recon.": "reconstructed", 
        "Smiling": "smiling", 
        "Sad": "sad", 
        "Angry": "angry", 
        "Man↔Woman": "gender", 
        "Young": "young", 
        "Curly hair": "curly_hair"
    }

    img_cols = []
    for attr in attrs:
        img_col = []
        for i, img_idx in enumerate(img_ids):
            if attrs[attr] == "original":
                img_path = f"{RUNSPATH}/smiling_{dt_lambda}_LC_CelebA_HQ_t999_ninv40_ngen40/test_images/40/original/test_{img_idx}_0_ngen40_original.png"
            elif attrs[attr] == "reconstructed":
                img_path = f"{RUNSPATH}/smiling_{dt_lambda}_LC_CelebA_HQ_t999_ninv40_ngen40/test_images/40/reconstructed/test_{img_idx}_0_ngen40_reconstructed.png"
            elif attrs[attr] == "gender":
                img_path = f"{RUNSPATH}/{man_woman[i]}_{dt_lambda}_LC_CelebA_HQ_t999_ninv40_ngen40/test_images/40/edited/test_{img_idx}_0_ngen40_edited.png"
            else:
                img_path = f"{RUNSPATH}/{attrs[attr]}_{dt_lambda}_LC_CelebA_HQ_t999_ninv40_ngen40/test_images/40/edited/test_{img_idx}_0_ngen40_edited.png"

            img = np.array(Image.open(img_path).convert('RGB'))
            img_col.append(img)

        img_cols.append(np.concatenate(img_col, axis=0))

    img_cols.insert(1, 255*np.ones((img_cols[0].shape[0], whitespace, img_cols[0].shape[2]), dtype=np.uint8))
    img_grid = np.concatenate(img_cols, axis=1)

    plt.rcParams.update({'font.size': 6})
    fig, ax = plt.subplots(1)

    xticks = [128] + [i*256 + 128 + whitespace for i in np.arange(1, len(img_cols)-1)]
    
    ax.set_xticks(xticks, tuple(attrs.keys()))
    ax.xaxis.tick_top()
    ax.xaxis.set_ticks_position('none')

    ax.set_yticks([])
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.imshow(img_grid)
    plt.savefig(f"{FIGURESPATH}/reproduction/in_{dt_lambda}.png", bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.show()

def reproduction_unseen(img_ids, dt_lambda=0.5, whitespace=30):
    """Function to reproduce samples using unseen-domain attributes.

    Args:
        img_ids (list): List of image IDs.
    """
    attrs = {
        "Real image": "original", 
        "Nicolas Cage": "nicolas", 
        "Pixar": "pixar", 
        "Modigliani": "modigliani", 
        "Neanderthal": "neanderthal", 
        "Frida": "frida"
    }

    img_cols = []
    for attr in attrs:
        img_col = []
        for i, img_idx in enumerate(img_ids):
            if attrs[attr] == "original":
                img_path = f"{RUNSPATH}/nicolas_{dt_lambda}_LC_CelebA_HQ_t999_ninv40_ngen40/test_images/40/original/test_{img_idx}_0_ngen40_original.png"
            else:
                img_path = f"{RUNSPATH}/{attrs[attr]}_{dt_lambda}_LC_CelebA_HQ_t999_ninv40_ngen40/test_images/40/edited/test_{img_idx}_0_ngen40_edited.png"

            img = np.array(Image.open(img_path).convert('RGB'))
            img_col.append(img)

        img_cols.append(np.concatenate(img_col, axis=0))

    img_cols.insert(1, 255*np.ones((img_cols[0].shape[0], whitespace, img_cols[0].shape[2]), dtype=np.uint8))
    img_grid = np.concatenate(img_cols, axis=1)

    plt.rcParams.update({'font.size': 6})
    fig, ax = plt.subplots(1)

    xticks = [128] + [i*256 + 128 + whitespace for i in np.arange(1, len(img_cols)-1)]

    ax.set_xticks(xticks, tuple(attrs.keys()))
    ax.xaxis.tick_top()
    ax.xaxis.set_ticks_position('none')

    ax.set_yticks([])
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.imshow(img_grid)
    plt.savefig(f"{FIGURESPATH}/reproduction/unseen_{dt_lambda}.png", bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.show()

def reproduction_linearity(img_id, start=-1.0, end=1.0, whitespace=30):
    img_path = f"{RUNSPATH}/smiling_interpolation_LC_CelebA_HQ_t999_ninv40_ngen40/test_images/40/test_{img_id}_0_ngen40.png"

    try:
        old_grid = np.array(Image.open(img_path).convert('RGB'))
    except:
        LookupError(f"The path {img_path} does not exist!\nInterpolations were only calculated for image IDs from 94 to 99.")

    imgs = deconstruct_grid(old_grid)
    imgs.insert(1, 255*np.ones((imgs[0].shape[0], whitespace, imgs[0].shape[2]), dtype=np.uint8))

    img_grid = np.concatenate(imgs, axis=1)

    # plt.rcParams.update({'font.size': 6})
    fig, ax = plt.subplots(1)

    xticks = [128] + [i*256 + 128 + whitespace for i in np.arange(1, len(imgs)-1)]
    xticks_names = ["Real image"] + [str(round(i, 2)) for i in np.linspace(start, end, len(imgs)-2)]
    
    if (len(imgs) % 2) == 1:
        xticks_names[int(len(imgs)/2)] = "Repr."

    ax.set_xticks(xticks, tuple(xticks_names))
    ax.xaxis.set_ticks_position('none')

    ax.set_yticks([])
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    #arrow_y = imgs[0].shape[0] + 140
    arrow_y = -20
    arrow_start = xticks[int(len(imgs)/2)]
    arrow_end1 = xticks[1]-xticks[int(len(xticks)/2)]
    arrow_end2 = xticks[-1]-xticks[int(len(imgs)/2)]

    ax.arrow(arrow_start, arrow_y, arrow_end1, 0, width=1, head_width=20, clip_on=False, color="black")
    ax.annotate('- Smiling (untrained)', 
                xy=(0,0), xytext=(arrow_start+arrow_end1/2, arrow_y-10), 
                xycoords='data', ha='center', annotation_clip=False)

    ax.arrow(arrow_start, arrow_y, arrow_end2, 0, width=1, head_width=20, clip_on=False, color="black")
    ax.annotate('+ Smiling (trained)', 
                xy=(0,0), xytext=(arrow_start+arrow_end2/2, arrow_y-10), 
                xycoords='data', ha='center', annotation_clip=False)

    plt.imshow(img_grid)

    plt.savefig(f"{FIGURESPATH}/reproduction/linearity.png", bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.show()

def reproduction_details(img_ids):
    img_cols = []
    for i, img_idx in enumerate(img_ids):
        img_col = []
        
        original_path = f"{RUNSPATH}/smiling_1.0_LC_CelebA_HQ_t999_ninv40_ngen40/test_images/40/original/test_{img_idx}_0_ngen40_original.png"
        edited_40_path = f"{RUNSPATH}/smiling_1.0_LC_CelebA_HQ_t999_ninv40_ngen40/test_images/40/edited/test_{img_idx}_0_ngen40_edited.png"
        edited_1000_path = f"{RUNSPATH}/smiling_1.0_LC_CelebA_HQ_t999_ninv40_ngen40/test_images/1000/edited/test_{img_idx}_0_ngen40_edited.png"

        original_img = np.array(Image.open(original_path).convert('RGB'))
        edited_40_img = np.array(Image.open(edited_40_path).convert('RGB'))
        edited_1000_img = np.array(Image.open(edited_1000_path).convert('RGB'))

        img_col.append(original_img)
        img_col.append(edited_40_img)
        img_col.append(edited_1000_img)

        img_cols.append(np.concatenate(img_col, axis=0))

    img_grid = np.concatenate(img_cols, axis=1)

    plt.rcParams.update({'font.size': 6})
    fig, ax = plt.subplots(1)
    
    ax.set_xticks([])

    yticks = [i*256 + 128 for i in np.arange(3)]

    ax.set_yticks(yticks, ("Real image", "40 steps", "1000 steps"))
    ax.yaxis.set_ticks_position('none')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.imshow(img_grid)
    plt.savefig(f"{FIGURESPATH}/reproduction/details.png", bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.show()

def reproduction_bias(img_ids):
    img_paths = [
        f"{RUNSPATH}/pixar_0.5_LC_CelebA_HQ_t999_ninv40_ngen40/bias/epoch4_h8_pixar/test_1_19_ngen40.png",
        f"{FIGURESPATH}/ablation/bias/epoch4_h8_pixar/test_21_19_ngen40.png"
    ]

    img_rows = []
    for i, img_idx in enumerate(img_ids):
        img_row = []
        
        original_path = f"{RUNSPATH}/pixar_0.5_LC_CelebA_HQ_t999_ninv40_ngen40/test_images/40/original/test_{img_idx}_0_ngen40_original.png"
        reconstructed_path = f"{RUNSPATH}/pixar_0.5_LC_CelebA_HQ_t999_ninv40_ngen40/test_images/40/reconstructed/test_{img_idx}_0_ngen40_reconstructed.png"
        edited_path = f"{RUNSPATH}/pixar_0.5_LC_CelebA_HQ_t999_ninv40_ngen40/test_images/40/edited/test_{img_idx}_0_ngen40_edited.png"

        original_img = np.array(Image.open(original_path).convert('RGB'))
        reconstructed_img = np.array(Image.open(reconstructed_path).convert('RGB'))
        edited_img = np.array(Image.open(edited_path).convert('RGB'))

        img_row.append(original_img)
        img_row.append(reconstructed_img)
        img_row.append(edited_img)

        img_rows.append(np.concatenate(img_row, axis=1))

    img_grid = np.concatenate(img_rows, axis=0)

    plt.rcParams.update({'font.size': 6})
    fig, ax = plt.subplots(1)

    xticks = [i*256 + 128 for i in np.arange(3)]
    
    ax.set_xticks(xticks, ("Real image", "Recon.", "Pixar"))
    ax.xaxis.tick_top()
    ax.xaxis.set_ticks_position('none')

    ax.set_yticks([])
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.imshow(img_grid)
    plt.savefig(f"{FIGURESPATH}/reproduction/bias.png", bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.show()


############
# ABLATION #
############

def ablation_epochs_vs_heads(reconstructed=False, image_idx=0):
    basepath = Path("/home/parting/master_AI/DL2/DL2-2023-group-15/src/eval_runs/heads_ablation")

    nheads = [1,2,4,8]
    epochs = [0,1,2,3,4,5]

    image_rows = []
    for nhead in nheads:
        image_row = []
        for epoch in epochs:
            if reconstructed:
                image_path = basepath / f"h{nhead}" / f"{epoch}" / "40" / "reconstructed" / f"test_{image_idx}_19_ngen40_reconstructed.png"
            else:
                image_path = basepath / f"h{nhead}" / f"{epoch}" / "40" / "edited" / f"test_{image_idx}_19_ngen40_edited.png"

            # read image
            image_array = np.array(Image.open(image_path).convert('RGB'))
            # if epoch == 0:
            #     image_array[:100] = 0

            image_row.append(image_array)
        row = np.concatenate(image_row, axis=0)
        image_rows.append(row)
    image_grid = np.concatenate(image_rows, axis=1)

    fig,ax = plt.subplots(1)
    ax.set_xticks([256] * np.arange(4) + 128, ("1","2","4","8"))
    ax.set_yticks([256] * np.arange(6) + 128, ("1","2","3","4", "5", "6"))
    ax.set_ylabel('epochs')
    ax.set_xlabel('nheads')
    plt.imshow(image_grid)
    if reconstructed:
        plt.savefig(f"epochs_vs_heads_img{image_idx}_recon.png", bbox_inches='tight', pad_inches=0, dpi=300)
    else:    
        plt.savefig(f"epochs_vs_heads_img{image_idx}.png", bbox_inches='tight', pad_inches=0, dpi=300)
    plt.show()

def ablation_epochs_vs_deltas(reconstructed=False, image_idx=0):
    basepath = Path("/home/parting/master_AI/DL2/DL2-2023-group-15/src/eval_runs/dstrength_ablation")

    nheads = [1,2,4,8]

    image_cols = []
    for nhead in nheads:
        image_path = basepath / f"h{nhead}" / "40" / f"test_{image_idx}_19_ngen40.png"

        # read image
        image_array = np.array(Image.open(image_path).convert('RGB'))

        image_cols.append(image_array)

    image_grid = np.concatenate(image_cols, axis=1)

    fig,ax = plt.subplots(1)
    ax.set_xticks([256] * np.arange(4) + 128, ("1","2","4","8"))
    # from script
    min_delta = 0
    max_delta = 1
    num_delta = 7
    strengths = np.linspace(min_delta, max_delta, num_delta)
    strengths = [f"{s:.3}" for s in strengths]
    ax.set_yticks([256] * np.arange(7) + 128, strengths)
    ax.set_ylabel('delta strength')
    ax.set_xlabel('nheads')
    plt.imshow(image_grid)
    plt.savefig(f"dstrength_vs_heads_img{image_idx}.png", bbox_inches='tight', pad_inches=0, dpi=300)
    # plt.show()

def ablation_epochs_vs_layer(reconstructed=False, image_idx=0):
    basepath = Path("/home/parting/master_AI/DL2/DL2-2023-group-15/src/eval_runs/layertype_ablation")

    layertype = ['pc_transformer_simple', 'cp_transformer_simple', 'c_transformer_simple', 'p_transformer_simple']
    # epochs = [0,1,2,3,4,5]
    epochs = [0,1]

    image_rows = []
    for layer in layertype:
        image_row = []
        for epoch in epochs:
            image_path = basepath / f"{layer}" / f"{epoch}" / "40" / "edited" / f"test_{image_idx}_19_ngen40_edited.png"

            # read image
            image_array = np.array(Image.open(image_path).convert('RGB'))
            # if epoch == 0:
            #     image_array[:100] = 0

            image_row.append(image_array)
        row = np.concatenate(image_row, axis=0)
        image_rows.append(row)
    image_grid = np.concatenate(image_rows, axis=1)

    fig,ax = plt.subplots(1)
    ax.set_xticks([256] * np.arange(4) + 128, layertype)
    ax.set_yticks([256] * np.arange(6) + 128, ("1","2","3","4", "5", "6"))
    ax.set_ylabel('epochs')
    ax.set_xlabel('layer type')
    
    plt.imshow(image_grid)
    plt.savefig(f"epochs_vs_layer_img{image_idx}.png", bbox_inches='tight', pad_inches=0, dpi=300)

    plt.show()
