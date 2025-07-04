from config import get_final_model
import argparse
import matplotlib.pyplot as plt
from pager.page_model.sub_models.converters import PDF2Img
from pager.page_model.sub_models.pdf_model import PDFModel
from pager.page_model.sub_models.image_model import ImageModel
from pager.page_model.sub_models.dtype import ImageSegment

model = get_final_model()
pdf = PDFModel()
img = ImageModel()
pdf2img = PDF2Img()


parser = argparse.ArgumentParser()
parser.add_argument('-i', type=str, nargs='?', required=True)
args = parser.parse_args()

def plot_file(path_pdf, class_=False):
    model.read_from_file(path_pdf)
    pdf.read_from_file(path_pdf)
    pdf2img.convert(pdf, img)
    model.extract()

    fig, (ax1, ax2)= plt.subplots(1, 2,dpi=200)
    ax1.set_xticks([])  # Remove x-axis ticks
    ax1.set_yticks([])  # Remove y-axis ticks
    plt.subplot(1, 2, 1)
    img.show()

    A = model.page_units[0].sub_model.json["A"]
    words = model.page_units[0].sub_model.json["words"]
    segs = [ImageSegment(dict_2p=w['segment']) for w in words]
    # for seg in segs:
    #     seg.plot()
    k = model.page_units[2].converters['json_model'].tmp_edge
    
    for r, e1, e2 in zip(k,  A[0], A[1]):
        b1 = segs[e1]
        b2 = segs[e2]
        x0, y0 = b1.get_center()
        x1, y1 = b2.get_center()
        plt.plot([x0, x1], [y0, y1], "g" if r > 0.5 else "r")

    plt.subplot(1, 2, 2)
    ax2.set_xticks([])  # Remove x-axis ticks
    ax2.set_yticks([])  # Remove y-axis ticks
    img.show()
    
    model.page_units[-1].sub_model.show(with_label=class_)

plot_file(args.i)
plt.show()