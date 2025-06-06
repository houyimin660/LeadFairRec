import pandas as pd
import json

def generate_user_books_json(dataset, file_path, output_file):
    # 交互信息
    inter = dataset.inter_feat

    # 反转字典
    field2token_id_user = dataset.field2token_id["user_id"]
    field2token_id_item = dataset.field2token_id["ISBN"]
    id_to_user = {v: k for k, v in field2token_id_user.items()}
    id_to_isbn = {v: k for k, v in field2token_id_item.items()}

    # 新增一列，将 user_id 转换为字典的 key
    inter['user_original'] = inter['user_id'].map(id_to_user)
    inter['isbn_original'] = inter['ISBN'].map(id_to_isbn)

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
        inter,
        book_item,
        on="isbn_original",
        how="left"  # 以 inter 的数据为主，保留未匹配的行
    )

    # 重命名 ISBN 列
    merged_df.rename(columns={
        "isbn_original": "Book_ISBN",  # 将 isbn_original 改为 book_isbn
    }, inplace=True)

    # 生成用户记录的字典格式
    user_data = []
    for user_id, group in merged_df.groupby('user_id'):
        books = group[['Book_ISBN', 'rating', 'title', 'author', 'public_year', 'publisher']].to_dict(orient='records')
        user_record = {"user_id": str(user_id), "books": books}
        user_data.append(user_record)

    # 将结果写入到 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in user_data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

# 调用函数
# 假设 dataset 是你传入的 dataset 对象，file_path 是 Book-Crossing.item 的路径
# output_file 是你希望输出的 JSON 文件路径
#generate_user_books_json(dataset, "dataset/Book-Crossing/Book-Crossing.item", 'llm_atributes/user_books_prompt.json')
