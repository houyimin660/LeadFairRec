import pandas as pd
import json
file_path="D:/study/Ada2Fair-main/dataset/Book-Crossing/Book-Crossing.item"
def generate_item_json(dataset, file_path, output_file):
    # 交互信息
    item = dataset.item_feat
    field2token_id_item = dataset.field2token_id["ISBN"]
    id_to_isbn = {v: k for k, v in field2token_id_item.items()}
    item['isbn_original'] = item['ISBN'].map(id_to_isbn)
    #rename for merge
    item.rename(columns={
        "ISBN":"item_id"# 将 isbn_original 改为 book_isbn
    }, inplace=True)
    item=item[['item_id','isbn_original']]
    # 读取文件到 DataFrame
    book_item = pd.read_csv(
        file_path,
        sep="\t",  # 使用制表符分隔
        header=0,  # 第一行为表头
        names=["isbn_original", "title", "author", "public_year", "publisher"],  # 指定列名
        dtype={"isbn_original": str, "title": str, "author": str, "public_year": str, "publisher": str}  # 确保数据类型
    )
    # 合并两个列表
    merged_df = pd.merge(
        item,
        book_item,
        on="isbn_original",
        how="left"  # 以 inter 的数据为主，保留未匹配的行
    )
    # 重命名 ISBN 列
    merged_df.to_json(output_file, orient='records', lines=True, force_ascii=False)