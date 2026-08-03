from model.model_embedder import embedder


class QueryProcessor:

    def process(self, query_text):

        cleaned_text = query_text.strip()

        "Ở đây có thể xứ lí thêm vài bước chuẩn hóa nữa"

        query_vector = embedder.encode_text(cleaned_text)

        return {

            "query_text": cleaned_text,

            "query_vector": query_vector

        }