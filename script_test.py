from config import PATH_TEST_DATASET, PATH_TEST_IMAGES, PATH_TEST_JSON, get_final_model 
from pager.benchmark.seg_detection.seg_detection_torchmetrics import  SegDetectionBenchmark


page = get_final_model()
benchmark = SegDetectionBenchmark(path_dataset=PATH_TEST_DATASET,
                                  page_model=page,
                                  path_images=PATH_TEST_IMAGES, 
                                  path_json=PATH_TEST_JSON)