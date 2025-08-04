import sys
from logging import getLogger
import argparse
from config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, init_seed, set_color
import os
import json
import pickle
from utils import *
from scipy.sparse import coo_matrix, csr_matrix
from utils import get_model
from trainer import Trainer

from llm_atributes.generate_age import *


def run_model(model_name, dataset_name, fairness_type):

    props = [
        "props/overall.yaml",
        f"props/{dataset_name}.yaml",
        f"props/{model_name}.yaml",
    ]
    print(props)

    model_class = get_model(model_name)

    # configurations initialization
    config = Config(
        model=model_class,
        dataset=dataset_name,
        config_file_list=props,
        config_dict={"fairness_type": fairness_type},
    )
    init_seed(config["seed"], config["reproducibility"])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    #sentmatic information
    usrprf_embeds_path = f"D:/study/Ada2Fair-main/llm_atributes/{dataset_name}/usr_emb_np.pkl"
    with open(usrprf_embeds_path, 'rb') as f:
        usrprf_embeds = pickle.load(f)

    itemprf_embeds_path=f"D:/study/Ada2Fair-main/llm_atributes/{dataset_name}/item_emb_np.pkl"
    with open(itemprf_embeds_path, 'rb') as f:
        itemprf_embeds = pickle.load(f)

    #sensitive add
    user_profile_file_beer='D:/study/Ada2Fair-main/llm_atributes/BeerAdvocate/user_profiles.json'
    age,age_mean=generate_age(user_profile_file_beer)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    #simlarity user select

    item_provider = dataset.get_item_feature()[
        config["PRODIVER_ID_FIELD"]
    ]

    interaction_matrix=dataset.inter_matrix(form="coo").astype(np.float32)

    user_item_csr = interaction_matrix.tocsr()

    provider_ids = item_provider.unique().numpy()
    provider_mapping = {pid: idx for idx, pid in enumerate(provider_ids)}
    num_providers = len(provider_ids)

    mapped_provider_ids = torch.tensor([provider_mapping[int(item)] for item in item_provider])
    #mapped_provider_ids = item_provider.map(provider_mapping).to_numpy()

    rows, cols, data = [], [], []
    for i in range(user_item_csr.shape[0]):
        user_row = user_item_csr.getrow(i).tocoo()
        for item_idx, interaction in zip(user_row.col, user_row.data):
            provider_idx = mapped_provider_ids[item_idx]
            rows.append(i)
            cols.append(provider_idx)
            data.append(interaction)
    # user_provider_matrix = coo_matrix(
    #     (data, (rows, cols)), shape=(6401, num_providers), dtype=np.float32
    # )
    user_provider_matrix = coo_matrix(
        (data, (rows, cols)), shape=(dataset.user_num, num_providers), dtype=np.float32
    )
    sim_users = calculate_user_similarity(user_provider_matrix, method="cosine", top_n=30)[0]

    #sim_providers=calculate_user_similarity(user_provider_matrix.T, method="cosine", top_n=30)[0]


    # model loading and initialization
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = model_class(config, train_data._dataset,sim_users,usrprf_embeds,itemprf_embeds,age).to(config["device"])

    logger.info(model)

    # trainer loading and initialization
    trainer = Trainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config["show_progress"]
    )

    # model evaluation
    test_result = trainer.evaluate(
        test_data, load_best_model=True, show_progress=config["show_progress"]
    )

    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")

    return (
        model_name,
        dataset_name,
        {
            "best_valid_score": best_valid_score,
            "valid_score_bigger": config["valid_metric_bigger"],
            "best_valid_result": best_valid_result,
            "test_result": test_result,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="BPR", help="name of models")
    parser.add_argument(
        "--dataset", "-d", type=str, default="Book-Crossing", help="name of datasets"
    )

    args, _ = parser.parse_known_args()

    run_model(args.model, args.dataset)
