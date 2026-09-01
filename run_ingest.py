from ingest.ingest_pipeline import IngestPipeline

if __name__ == "__main__":
    print("=====本地测试Ingest流水线，处理./data目录=====")
    pipeline = IngestPipeline()
    pipeline.process_dir("./data")
