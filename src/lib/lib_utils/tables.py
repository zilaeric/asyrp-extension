import json

def get_b(content):
    return f"<b>{content}</b>"

def get_td(content, loc=None):
    if loc:
        return f"\t\t<td align=\"{loc}\">{content}</td>\n"
    else:
        return f"\t\t<td>{content}</td>\n"

def get_th(content, loc=None):
    if loc:
        return f"\t\t<th align=\"{loc}\">{content}</th>\n"
    else:
        return f"\t\t<th>{content}</th>\n"

def get_tr(arr, loc=None, header=False):
    tag_fn = get_th if header else get_td

    content = tag_fn(arr[0], "left")
    for i in range(1, len(arr)):
        content += tag_fn(arr[i])

    if loc:
        return f"\t<tr align=\"{loc}\">\n{content}\t</tr>\n"
    else:
        return f"\t<tr>\n{content}\t</tr>\n"

def get_tr_caption(id, desc, cols, loc="center"):
    tr = f"\t<tr align=\"{loc}\">\n"
    tr += f"\t\t<td colspan={cols}><b>{id}.</b> {desc}</td>\n"
    tr += "\t</tr>\n"

    return tr

def get_table(header, contents, id, desc):
    table = "<table align=\"center\">\n"
    table += get_tr(header, "center", "true")

    for content in contents:
        table += get_tr(content, "center")

    table += get_tr_caption(id, desc, len(header), "left")
    table += "</table>"

    return table


################
# REPRODUCTION #
################

def reproduction_sdir():
    header = [
        "Metric", "", "Smiling (IN)", 
        "Sad (IN)", "Tanned (IN)", 
        "Pixar (UN)", "Neanderthal (UN)"
    ]

    attrs = ["smiling", "sad", "tanned", "pixar", "neanderthal"]

    with open("reproduction_sdir_1.0.json") as f:
        full_dict = json.load(f)

    with open("reproduction_sdir_0.5.json") as f:
        half_dict = json.load(f)

    original = ["Original $S_{dir}$", "$\Delta h_t$"] + [0.921, 0.964, 0.991, 0.956, 0.805]
    full = ["Reproduced $S_{dir}$", "$\Delta h_t$"] + [format(1-full_dict[attr]["sdir_oe"], ".3f") + "<br>(" + format(full_dict[attr]["sdir_oe_var"], ".3f") + ")" for attr in attrs]
    half = ["Reproduced $S_{dir}$", "$0.5 \Delta h_t$"] + [format(1-half_dict[attr]["sdir_oe"], ".3f") + "<br>(" + format(half_dict[attr]["sdir_oe_var"], ".3f") + ")" for attr in attrs]

    id = "Table 2"
    desc = "Directional CLIP score ($S_{dir}$) for in-domain (IN) and unseen-domain (UN) attributes. Standard<br>deviations are reported in parentheses."

    html = get_table(header, [original, full, half], id, desc)

    print(html)

def reproduction_fid():
    header = [
        "Metric", "", "Smiling (IN)", 
        "Sad (IN)", "Tanned (IN)", 
        "Pixar (UN)", "Neanderthal (UN)"
    ]

    attrs = ["smiling", "sad", "tanned", "pixar", "neanderthal"]

    with open("reproduction_fid_1.0.json") as f:
        full_dict = json.load(f)

    with open("reproduction_fid_0.5.json") as f:
        half_dict = json.load(f)
    
    full_eo = ["$FID(\mathbf{x}_{orig}, \mathbf{x}_{edit})$", "$\Delta h_t$"] + [format(full_dict[attr]["score_eo"], ".1f") for attr in attrs]
    full_er = ["$FID(\mathbf{x}_{recon}, \mathbf{x}_{edit})$", "$\Delta h_t$"] + [format(full_dict[attr]["score_er"], ".1f") for attr in attrs]

    half_eo = ["$FID(\mathbf{x}_{orig}, \mathbf{x}_{edit})$", "$0.5 \Delta h_t$"] + [format(half_dict[attr]["score_eo"], ".1f") for attr in attrs]
    half_er = ["$FID(\mathbf{x}_{recon}, \mathbf{x}_{edit})$", "$0.5 \Delta h_t$"] + [format(half_dict[attr]["score_er"], ".1f") for attr in attrs]

    id = "Table 3"
    desc = "Frechet Inception Distance ($FID$) for in-domain (IN) and unseen-domain (UN) attributes."

    html = get_table(header, [full_eo, half_eo, full_er, half_er], id, desc)

    print(html)

def reproduction_fid_bias():
    header = [
        "Metric", "Race", "Smiling (IN)", 
        "Sad (IN)", "Tanned (IN)", 
        "Pixar (UN)", "Neanderthal (UN)"
    ]

    attrs = ["smiling", "sad", "tanned", "pixar", "neanderthal"]

    with open("reproduction_fid_1.0_caucasian.json") as f:
        caucasian_dict = json.load(f)

    with open("reproduction_fid_1.0_noncaucasian.json") as f:
        non_dict = json.load(f)
    
    caucasian_eo = ["$FID(\mathbf{x}_{orig}, \mathbf{x}_{edit})$", "Caucasian"] + [format(caucasian_dict[attr]["score_eo"], ".1f") for attr in attrs]
    caucasian_er = ["$FID(\mathbf{x}_{recon}, \mathbf{x}_{edit})$", "Caucasian"] + [format(caucasian_dict[attr]["score_er"], ".1f") for attr in attrs]

    no_eo = ["$FID(\mathbf{x}_{orig}, \mathbf{x}_{edit})$", "Non-caucasian"] + [format(non_dict[attr]["score_eo"], ".1f") for attr in attrs]
    no_er = ["$FID(\mathbf{x}_{recon}, \mathbf{x}_{edit})$", "Non-caucasian"] + [format(non_dict[attr]["score_er"], ".1f") for attr in attrs]

    id = "Table 4"
    desc = "Frechet Inception Distance ($FID$) for in-domain (IN) and unseen-domain (UN) attributes compared between caucasian and non-caucasian individuals."

    html = get_table(header, [caucasian_eo, no_eo, caucasian_er, no_er], id, desc)

    print(html)


############
# ABLATION #
############

def ablation_fid_epochs():
    header = ["Metric"] + ["Epoch " + str(i+1) for i in range(4)]

    with open("metrics.json") as f:
        metrics = json.load(f)
    
    pc_eo = ["$FID(\mathbf{x}_{orig}, \mathbf{x}_{edit})$"]
    pc_er = ["$FID(\mathbf{x}_{recon}, \mathbf{x}_{edit})$"]

    for i in range(4):
        pc_eo += [format(metrics[f"layertype_ablation/pc_transformer_simple_{i}_pixar"]["fid_edited_original"], ".1f")]
        pc_er += [format(metrics[f"layertype_ablation/pc_transformer_simple_{i}_pixar"]["fid_edited_reconstructed"], ".1f")]

    id = "Table 5"
    desc = "Frechet Inception Distance ($FID$) with pixel-channel <br> architecture for the \"pixar\" attribute across epochs."

    html = get_table(header, [pc_eo, pc_er], id, desc)

    print(html)

def ablation_fid():
    header = [
        "Model", "Smiling (IN)", 
        "Sad (IN)", "Tanned (IN)", 
        "Pixar (UN)", "Neanderthal (UN)"
    ]

    attrs = ["smiling", "sad", "tanned", "pixar", "neanderthal"]

    with open("../reproduction/reproduction_fid_1.0.json") as f:
        reproduction_dict = json.load(f)

    with open("metrics_fid_table.json") as f:
        ablation_dict = json.load(f)
    
    reproduction_eo_full = ["Original"] + [format(reproduction_dict[attr]["score_eo"], ".1f") for attr in attrs]
    
    ablation_eo_full = ["Ours"]
    for attr in attrs:
        ablation_eo_full += [format(ablation_dict[f"fid_table_pc_h8{attr}_LC_CelebA_HQ_t999_ninv40_ngen40/test_imag_s_{attr}"]["fid_edited_original"], ".1f")]

    id = "Table 5"
    desc = "Comparison of Frechet Inception Distance ($FID \downarrow$) metric for in-domain (IN) and unseen-domain (UN) attributes between the original model and our best model."

    html = get_table(header, [reproduction_eo_full, ablation_eo_full], id, desc)

    print(html)

