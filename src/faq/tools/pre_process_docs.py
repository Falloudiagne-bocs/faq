import os
import uuid
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

from openai import OpenAI
from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter
from minio import Minio
from minio.error import S3Error
from datetime import timedelta

# === Configuration environnement ===
load_dotenv()

COLLECTION_NAME = "decret-docs-crewai"
embedding_model = "text-embedding-3-small"
vector_size = 1536

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    prefer_grpc=False
)



# Configuration MinIO
client = Minio(
    os.getenv("ENDPOINT"),
    access_key=os.getenv("ACCESS_KEY"),
    secret_key=os.getenv("SECRET_KEY"),
    secure=False
)

bucket_name = "iachatbotbocs"

pdf_folder = "/home/user/projects/crewai/Chatbot/faq/knowledge/pdfs"

# Télécharger tous les PDF du dossier 'decret' du bucket
decret_prefix = "decret/"
pdf_objects = client.list_objects(bucket_name, prefix=decret_prefix, recursive=True)
local_decret_dir = pdf_folder
os.makedirs(local_decret_dir, exist_ok=True)

for obj in pdf_objects:
    if obj.object_name.endswith(".pdf"):
        local_path = os.path.join(local_decret_dir, os.path.basename(obj.object_name))
        if not os.path.exists(local_path):
            client.fget_object(bucket_name, obj.object_name, local_path)


# === Préparation des documents ===
doc_converter = DocumentConverter(allowed_formats=[InputFormat.PDF])
chunker = HybridChunker()
points = []

# === Lecture des PDF et indexation ===
for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, filename)
        print(f"📄 Processing: {pdf_path}")

        result = doc_converter.convert(pdf_path)

        for chunk in chunker.chunk(result.document):
            embedding_result = openai_client.embeddings.create(
                input=[chunk.text],
                model=embedding_model
            )
            vector = embedding_result.data[0].embedding

            point_id = str(uuid.uuid4())
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "text": chunk.text,
                        "metadata": chunk.meta.export_json_dict(),
                    }
                )
            )

print(f"{len(points)} vecteurs prêts à être insérés dans Qdrant.")

# === Création de la collection ===
if COLLECTION_NAME in [c.name for c in qdrant_client.get_collections().collections]:
    qdrant_client.delete_collection(collection_name=COLLECTION_NAME)

qdrant_client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
)

qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
print(" Indexation terminée dans Qdrant.\n")

# === Recherche avec query_points ===
"""  
query = "What is the best to use for vector search scaling?" 
query_vector = openai_client.embeddings.create(
    input=[query],
    model=embedding_model
).data[0].embedding

results = qdrant_client.query_points(
    collection_name=COLLECTION_NAME,
    query_vector=query_vector,
    limit=10
)

# === Affichage des résultats ===
for i, point in enumerate(results, 1):
    print(f"\n🔍 Résultat {i} (score: {point.score:.4f})")
    print(point.payload.get("text", "")[:300])  # Affiche un extrait du texte

"""