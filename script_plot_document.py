from config import get_final_model
import argparse
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()
parser.add_argument('-i', type=str, nargs='?', required=True)
args = parser.parse_args()


model = get_final_model()
model.read_from_file(args.i)
model.extract()
model.page_units[0].sub_model.show()
model.page_units[2].sub_model.show()
plt.show()