import os
import json
import tqdm
import argparse
import pandas as pd

from openai import OpenAI

"""
将从网上下载的公开数据集中的评论数据，调用大模型转换为情感分析数据集
输出内容为jsonl格式，每行内容为json格式，包含content、type、prediction三个字段
"""


class CommentSentimentTransformer(object):
    OPEN_AI_URL = os.getenv("OPEN_AI_URL", "https://api.openai.com/v1/chat/completions")
    OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY", "sk-b1caf00b08e242afb420831329ec0d00")
    MODEL = "Qwen2.5-72B-Instruct-AWQ"
    SYSTEM_PROMPT = """你是一个情感分析专家，请分析以下评论的情感倾向，需要根据用户提供的评论内容，对该评论进行分类，总共有六个等级。\n
                以下是针对商品评论设计的六级分类标签：\n
                1. **非常满意**： \n
                     - 表示用户对商品极其满意，可能会使用诸如“完美”、“超乎想象”、“非常推荐”等词语。\n
                     - 示例评论：“这款手机的性能太棒了，完全超出我的预期！”\n
                2. **满意**：\n
                     - 表示用户对商品满意，但可能还有改进的空间。评论中可能会提到一些小问题，但总体上是正面的。\n
                     - 示例评论：“这款耳机音质不错，佩戴也很舒适，只是电池续航稍短。”\n
                3. **中立**：\n
                     - 表示用户对商品没有明显的正面或负面情绪，可能是因为商品符合预期，但没有特别突出的优点或缺点。\n
                     - 示例评论：“这款洗发水用起来还可以，没有什么特别好的地方，也没有什么不好的地方。”\n
                4. **不满意**：\n
                     - 表示用户对商品有一些不满，可能遇到了一些问题或不符合预期。评论中通常会提到具体的问题点。\n
                     - 示例评论：“这款相机的对焦速度太慢了，拍运动场景经常模糊。”\n
                5. **非常不满意**：\n
                     - 表示用户对商品极度不满，可能会强烈批评产品，甚至建议其他人不要购买。\n
                     - 示例评论：“这款电视质量太差了，用了不到一个月就坏了，售后也不好，绝对不推荐！”\n
                6. **无效评论**：\n
                     - 用户输入的内容语句不通顺或者无任何意义。\n
                     - 示例评论：“刚刚发愁肠百结健康刚刚过\n
                你只需要根据用户的评论内容，判断该评论属于哪个等级，不要输出多个等级，只输出一个等级即可,不需要输出原因。
                """

    def __init__(self, file_path, comment_column, type_column, output_path):
        self.client = OpenAI(base_url=self.OPEN_AI_URL, api_key=self.OPEN_AI_API_KEY)
        self.file_path = file_path
        self.comment_column = comment_column
        self.type_column = type_column
        self.llm_temperature = 0.5
        self.llm_max_tokens = 1024
        self.llm_top_p = 0.9
        self.output_path = output_path

    def load_data(self):
        """
        读取数据
        :param file_path: 文件路径
        :return: 数据列表
        """
        with open(self.file_path, 'r', encoding='utf-8') as f:
            data = pd.read_csv(self.file_path, sep='\t')
            # 转为字符串, 并去除空格    
            data[self.comment_column] = data[self.comment_column].apply(lambda x: str(x).strip())
            # 如果 type_column 为空, 则设置为 ""
            if self.type_column not in data.columns:
                data[self.type_column] = ""
            # 如果type列数据为空，则转为空字符串
            data[self.type_column] = data[self.type_column].apply(lambda x: "" if pd.isna(x) else x)
            self.data = data[[self.comment_column, self.type_column]].to_dict(orient='records')
    
    def get_sentiment(self, comment):
        """
        获取评论的情感倾向
        :param comment: 评论内容
        :return: 情感倾向
        """
        messages=[
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": comment}
        ]
        response = self.client.chat.completions.create(
            model=self.MODEL, messages=messages,
            temperature=self.llm_temperature, max_tokens=self.llm_max_tokens, top_p=self.llm_top_p
        )
        return response.choices[0].message.content    

    def run(self):
        """
        处理数据, 并保存jsonl格式
        """
        print(f"开始处理数据, 并保存到 {self.output_path}")
        self.load_data()
        # 记录当前处理到的行数
        processed_count = 0
        if os.path.exists("processed_count.txt"):
            with open("processed_count.txt", "r") as f:
                processed_count = max([int(line.strip()) for line in f.readlines() if line.strip().isdigit()])
                print(f"当前处理到的行数为 {processed_count}")
        with open(self.output_path, 'a', encoding='utf-8') as f1, open("processed_count.txt", "w") as f2:
                try:
                        for item in tqdm.tqdm(self.data[processed_count:]):
                                sentiment = self.get_sentiment(item[self.comment_column])
                                if sentiment not in ["非常满意", "满意", "中立", "不满意", "非常不满意", "无效评论"]:
                                    continue
                                json_data = {
                                "content": item[self.comment_column],
                                "type": item[self.type_column],
                                "prediction": sentiment
                                }
                                f1.write(json.dumps(json_data, ensure_ascii=False) + "\n")
                                processed_count += 1
                                f2.write(str(processed_count) + "\n")
                except Exception as e:
                        f2.write(str(processed_count) + "\n")
                        print(f"处理数据时发生错误: {e}")
        print(f"数据处理完成, 并保存到 {self.output_path}")


if __name__ == "__main__":
    # 使用示例 python dataset_trans.py -f data/comment.csv -c comment -t type -o data/comment_sentiment.jsonl
    parser = argparse.ArgumentParser(description="情感分析数据预处理")
    parser.add_argument("-f", "--file_path", type=str, help="输入文件路径")
    parser.add_argument("-c", "--comment_column",    type=str, help="评论列名")
    parser.add_argument("-t", "--type_column", type=str, help="类型列名")
    parser.add_argument("-o", "--output_path", type=str, help="输出文件路径")
    args = parser.parse_args()
    # 检查参数是否为空
    if not all([args.file_path, args.comment_column, args.type_column, args.output_path]):
        raise ValueError("所有参数均不可为空")
    comment_sentiment_transformer = CommentSentimentTransformer(
        file_path=args.file_path, comment_column=args.comment_column, type_column=args.type_column, 
        output_path=args.output_path
    )
    comment_sentiment_transformer.run()
