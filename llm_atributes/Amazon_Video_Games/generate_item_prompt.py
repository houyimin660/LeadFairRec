import pandas as pd
import json
file_path='D:/study/Ada2Fair-main/dataset/BeerAdvocate/BeerAdvocate.item'
def generate_item_json(dataset, file_path, output_file):
    # 交互信息
    item = dataset.item_feat
    field2token_id_item = dataset.field2token_id["item_id"]
    id_to_isbn = {v: k for k, v in field2token_id_item.items()}
    item['item_original'] = item['item_id'].map(id_to_isbn)
    item=item[['item_id','item_original']]
    # 读取文件到 DataFrame
    beer_item = pd.read_csv(
        file_path,
        sep="\t",  # 使用制表符分隔
        header=0,  # 第一行为表头
        names=["item_original", "name", "brewer_id", "ABV", "style"],  # 指定列名
        dtype={"item_original": str, "name": str, "brewer_id": str, "ABV": float, "style": str}  # 确保数据类型
    )
    # 合并两个列表
    merged_df = pd.merge(
        item,
        beer_item,
        on="isbn_original",
        how="left"  # 以 inter 的数据为主，保留未匹配的行
    )
    # 重命名 ISBN 列
    merged_df.to_json(output_file, orient='records', lines=True, force_ascii=False)