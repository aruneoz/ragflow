import asyncio
import copy
import random
from pathlib import Path
import nest_asyncio
import pandas as pd
from llama_index.core.evaluation import BatchEvalRunner
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.callbacks import CallbackManager
from llama_index.callbacks.wandb import WandbCallbackHandler
import wandb
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.llms.anthropic import Anthropic
# generate testset
from llama_index.core.llama_pack import download_llama_pack

model_name = "yolox"

REGION = 'us-central1' # @param ["us-central1", "asia-southeast1"]
ENDPOINT=f"https://{REGION}-aiplatform.googleapis.com"
PROJECT_ID = 'cloud-llm-preview1'  # @param {type: "string"} # cloud-llm-preview2
MODEL = 'claude-instant-1' # @param ["claude-2p0", "claude-instant-1p2"]

antropic_llm = Anthropic(base_url=ENDPOINT,model=MODEL,max_tokens=1024,temperature=0.1)


#client = anthropic.AnthropicVertex(region=REGION, project_id=PROJECT_ID)
# message = client.beta.messages.create(
#     max_tokens=1024,
#     messages=[
#         {
#             "role": "user",
#             "content": "Write a function that checks if a year is a leap year.",
#         }
#     ],
#     model=MODEL,  # required argument but isn't sent at runtime
# )
# print(message.model_dump_json(indent=2))

from llama_index.core.evaluation import DatasetGenerator
from llama_index.llms.vertex import Vertex
from llama_index.core import ServiceContext, VectorStoreIndex

from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    CorrectnessEvaluator,
)
from llama_index.core.evaluation import ResponseEvaluator

# wandb.login(key='d63079f331f1f9cb336ebdeb01008d4bf5fed759')
#
# wandb_args = {"project":"gemini-test"}
# wandb_callback = WandbCallbackHandler(run_args=wandb_args)

# callback_manager = CallbackManager([wandb_callback])

async def evaluate_questions(evaluate_dataset, evaluator):
    # Simulate some processing with I/O bound task
    result = evaluator.evaluate(
        query=evaluate_dataset.query,
        response=evaluate_dataset.reference_answer,
        reference=evaluate_dataset.reference_contexts[0],
    )
    return f"Processed: {result}"



async def batch():

    model_name = "models/embedding-001"
    embed_model = GeminiEmbedding(model_name=model_name,api_key='AIzaSyBbPtJIkAMPY3QuoM_ofe4wgdBuRKfyGzU')
    # loader = UnstructuredReader()
    # documents = loader.load_data(file=Path('../../data/rockwell_pdf_sample_10_pages.pdf'))
    from llama_index.core import SimpleDirectoryReader

    documents = SimpleDirectoryReader("../../data/pdf").load_data()
    vertexai_context = ServiceContext.from_defaults(embed_model=embed_model,
                                                    llm=Vertex(model="gemini-pro", temperature=0, additional_kwargs={})
                                                    )

    vertexai_antrophic_context = ServiceContext.from_defaults(embed_model=embed_model,
                                                    llm=antropic_llm
                                                    )
    # service_context_gemini = ServiceContext.from_defaults(embed_model=embed_model,
    #                                                       llm=Vertex(model="text-bison", temperature=0,callback_manager=callback_manager,
    #                                                                  additional_kwargs={}))

    service_context_gemini = ServiceContext.from_defaults(embed_model=embed_model,
                                                          llm=Vertex(model="text-bison", temperature=0,
                                                                     additional_kwargs={}))

    # gemini_pro_context = ServiceContext.from_defaults(embed_model=embed_model,
    #     llm=Gemini(model="models/gemini-pro",temperature=0,api_key='AIzaSyBbPtJIkAMPY3QuoM_ofe4wgdBuRKfyGzU',callback_manager=callback_manager)
    # )



    # Setting up the documents and generating questions for evaluation
    random_documents = copy.deepcopy(documents)

    # Shuffling the documents and selecting 5 random documents. Just to make the evaluation quicker
    random.shuffle(random_documents)
    random_documents = random_documents[:5]

    # # Generating questions from the documents for evaluation
    # data_generator = DatasetGenerator.from_documents(
    #     random_documents, service_context=vertexai_context, num_questions_per_chunk=2
    # )
    #
    # # Applying nest_asyncio to run async code in Jupyter
    # nest_asyncio.apply()
    # eval_questions = data_generator.generate_questions_from_nodes()
    from llama_index.core.llama_dataset.generator import RagDatasetGenerator
    from llama_index.core.prompts.base import PromptTemplate
    from llama_index.core.prompts.prompt_type import PromptType

    DEFAULT_TEXT_QA_PROMPT_TMPL = (
        "Context information is below. The document contains the user guide for Rockwell Automation systems\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, "
        "answer the query from the context_str \n"
        "Query: {query_str}\n"
        "Answer: "
    )
    DEFAULT_TEXT_QA_PROMPT = PromptTemplate(
        DEFAULT_TEXT_QA_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER
    )
    num_questions_per_chunk=2
    dataset_generator = RagDatasetGenerator.from_documents(
        random_documents,
        llm=Vertex(model="text-bison", temperature=0,additional_kwargs={}),
        num_questions_per_chunk=2,
        text_qa_template=DEFAULT_TEXT_QA_PROMPT,

        # question_gen_query=f"You are a Teacher/Professor. Your task is to setup \
        #                 {num_questions_per_chunk} questions for an upcoming \
        #                 quiz/examination. The questions should be diverse in nature \
        #                     across the document. Restrict the questions to the \
        #                         context information provided." ,# set the number of questions per nodes
        show_progress=True,
    )
    nest_asyncio.apply()
    #rag_dataset = dataset_generator.generate_dataset_from_nodes()
    eval_questions = dataset_generator.generate_questions_from_nodes()

    #print(eval_questions.to_pandas().info())
    question_df = eval_questions.to_pandas()
    #question_df.info()
    #question_df['query'] = question_df['query'].astype(str)
    #question_df.info()
    print(question_df['query'].iloc[1])
    print(question_df['reference_contexts'].iloc[1])
    print(question_df['reference_answer'].iloc[1])
    # #
    #rag_df_dataset=rag_dataset.to_pandas()
    #questions=rag_df_dataset['query'].tolist()
    #print(questions)
    eval_questions.save_json(f"rockwell_pdf_sample_10_pages_123.json")
    #

    faithfulness_evaluator = FaithfulnessEvaluator(service_context=vertexai_antrophic_context)
    relevancy_evaluator = RelevancyEvaluator(service_context=vertexai_antrophic_context)

    runner = BatchEvalRunner(
        {"faithfulness": faithfulness_evaluator, "relevancy": relevancy_evaluator},
        workers=8,
    )
    #
    # rag_dataset.to_pandas()
    #
    # print(rag_dataset.to_pandas().head(2))

    vector_store = PGVectorStore.from_params(
        database="pgvector",
        host="34.170.81.51",
        password="sUmmertime123$",
        port=5432,
        user="admin",
        table_name="neumai_vector_test",
        embed_dim=768,  # vertexai embedding dimension
    )

    # print(rag_dataset.examples[0].query)

    vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store, service_context=service_context_gemini)
    query_engine = vector_index.as_query_engine()
    from llama_index.core.llama_pack import download_llama_pack

    # prediction_dataset = await eval_questions.amake_predictions_with(
    #     predictor=query_engine, show_progress=True
    # )
    #
    # print(prediction_dataset.to_pandas()[:5])

    # RagEvaluatorPack = download_llama_pack("RagEvaluatorPack", "./pack")
    #
    # rag_evaluator = RagEvaluatorPack(
    #     query_engine=query_engine,  # built with the same source Documents as the rag_dataset
    #     rag_dataset=eval_questions,
    # )
    # benchmark_df = await rag_evaluator.run()
    # print(benchmark_df)
    # define evaluator
    # embed_model = GeminiEmbedding(model_name=model_name, api_key='AIzaSyBbPtJIkAMPY3QuoM_ofe4wgdBuRKfyGzU')
    # gemini_pro_context = ServiceContext.from_defaults(embed_model=embed_model,
    #                                                   llm=Gemini(model="models/gemini-pro", temperature=0,
    #                                                              api_key='AIzaSyBbPtJIkAMPY3QuoM_ofe4wgdBuRKfyGzU')
    #                                                   )
    #
    # query="What is the purpose of the Power Interface Module (PIM) in the ArmorKinetix system?"
    # reference = "About the ArmorKinetix System\n\nTable 2 - ArmorKinetix System Overview\n\nDrive System Component\n\nCat. No.\n\nDC-bus Power Supply\n\n2198-Pxxx\n\nArmorKinetix Power Interface Module (PIM)\n\n2198-PIM070\n\nArmorKinetix System Single-axis Distributed Servo Drives (DSD)\n\n2198-DSD0xx-ERS2 2198-DSD0xx-ERS5\n\nArmorKinetix System Single-axis Distributed Servo Motors (DSM)\n\n2198-DSM0xx-ERS2 2198-DSM0xx-ERS5\n\nKinetix\u00ae 5700 Capacitor Module\n\n2198-CAPMOD-2240\n\nKinetix 5700 Extension Module\n\n2198-CAPMOD-DCBUS-IO\n\nKinetix 5700 DC-bus Conditioner Module\n\n2198-DCBUSCOND-RP312\n\nShared-bus Connector Kits\n\n2198-TCON-24VDCIN36 2198-xxxx-P-T 2198-BARCON-xxDCAC100\n\n2198-BARCON-xxDC200 2198-KITCON-ENDCAP200\n\nPIM Connector Kit Kinetix 5700 System Mounting Toolkit\n\n2198-KITCON-PIM070\n\n2198-K5700-MOUNTKIT\n\nEncoder Output Module 2198-ABQE\n\nChapter 1\n\nStart\n\nUse this chapter to become familiar with the ArmorKinetix\u00ae system and obtain an overview of installation configurations.\n\nOn-machine drives include Distributed Servo Motor (DSM) and Distributed Servo Drive (DSD). Both are single axis inverters and can be powered by a Diode Front End (DFE) module. The connection between the in-cabinet system and the on-machine inverters is established by using the Power Interface Module (PIM) that distributes DC power and communication signals by using a single cable (hybrid cable). Each PIM module can support up to 24 axes. If more than 24 axes are needed, you can use multiple PIM modules.\n\nDescription\n\nConverter power supply with 200V and 400V-class (three-phase) AC input. Provides output current in a range of 10.5\u202669.2 A. Systems typically consist of one module, however, up to three modules in parallel is possible. Parallel modules increase available power for 2198 modules. The PIM module provides the connection between the in-cabinet system and the on-machine inverter. This module distributes DC power and communication signals to the DSD and DSM modules by using a single cable (hybrid cable). Each PIM module can support up to 24 axes. Single-axis inverters with current ratings up to 8 A rms. Drives feature T\u00dcV Rheinland-certified Safe Torque Off function with integrated safety connection options, PLe and SIL 3 safety ratings, and support for Hiperface DSL, and Hiperface encoder feedback. The DSD modules also support Timed and Monitored SS1 drive-based stopping functions, and support for controller based Safe Stop 1 and safe speed monitoring functions over the Ethernet network. Single-axis motor/inverter with maximum continuous torque of 11.9 Nm and peak torque of 31.2 Nm with speeds up to 8000 rpm. The -ERS2 motor/inverters feature T\u00dcV Rheinland-certified Safe Torque Off function with integrate safety connection, PLe, and SIL 3 for Hiperface DSL feedback only. The modules also support Timed SS1 drive-based stopping functions. The 2198-DSMxxx-ERS5 modules also support Timed and Monitored SS1 drive-based stopping functions, and support for controller-based Safe Stop 1 and Safely-limited Speed functions. Use for energy storage, external active-shunt connection, and to extend the DC-bus voltage to another inverter cluster. Modules are zero-stacked with servo drives and use the shared-bus connection system to extend the external DC-bus voltage in applications up to 104 A. Can parallel with itself or with another accessory module for up to 208 A with required 2198-KITCON-CAPMOD2240 kit that includes flexible bus-bars. The extension module, paired with a capacitor module or DC-bus conditioner module, is used to extend the DC-bus voltage to another inverter cluster in systems with \u2265104 A current and up to 208 A. Decreases the voltage stress on insulation components in an inverter system and used to extend the DC-bus voltage to another inverter cluster."
    # answer = "The Power Interface Module (PIM) in the ArmorKinetix system provides the connection between the in-cabinet system and the on-machine inverter. It distributes DC power and communication signals to the DSD and DSM modules by using a single cable (hybrid cable). Each PIM module can support up to 24 axes."
    # evaluators = {
    #     # "gpt-4": CorrectnessEvaluator(service_context=gpt_4_context),
    #     # "gpt-3.5": CorrectnessEvaluator(service_context=gpt_3p5_context),
    #     "gemini-pro": CorrectnessEvaluator(service_context=gemini_pro_context),
    # }
    evaluator = CorrectnessEvaluator(llm=vertexai_antrophic_context)
    #
    # # result = evaluator.evaluate(
    # #     query=eval_questions.examples[0].query,
    # #     response=eval_questions.examples[0].reference_answer,
    # #     reference=eval_questions.examples[0].reference_contexts[0],
    # # )
    #
    results = await asyncio.gather(*[evaluate_questions(eval_question,evaluator) for eval_question in eval_questions.examples])
    print(results)


    # eval_results = await runner.aevaluate_queries(
    #     query_engine=query_engine, queries=question_df['query'].tolist()
    # )
    # faithfulness_df = pd.DataFrame.from_records([eval_result.dict() for eval_result in eval_results["faithfulness"]])
    # relevancy_df = pd.DataFrame.from_records([eval_result.dict() for eval_result in eval_results["relevancy"]])
    # print(faithfulness_df)
    # # We need score!
    # print(faithfulness_df["score"].mean(), relevancy_df["score"].mean())

#     EvaluatorBenchmarkerPack = download_llama_pack("EvaluatorBenchmarkerPack", "./pack")
#     evaluator_benchmarker = EvaluatorBenchmarkerPack(evaluator=evaluators['gemini-pro'],
#     eval_dataset=rag_dataset,
#     show_progress=True,
# )
#
#     gemini_pro_benchmark_df = await evaluator_benchmarker.arun(
#         batch_size=5, sleep_time_in_seconds=0.5
#     )
#     gemini_pro_benchmark_df.index = ["gemini-pro"]
#     print(gemini_pro_benchmark_df.head(1))
# # produce the benchmark result
#     benchmark_df = await evaluator_benchmarker.arun(
# 		batch_size=5,
# 		sleep_time_in_seconds=0.5
#     )

    # result = evaluator.evaluate(
    #     query=rag_dataset.examples[0].query,
    #     response=context,
    #     context=context,
    # )

    #Make a dataframe from the results.
    # faithfulness_df = pd.DataFrame.from_records([eval_result.dict() for eval_result in eval_results["faithfulness"]])
    # relevancy_df = pd.DataFrame.from_records([eval_result.dict() for eval_result in eval_results["relevancy"]])

    # We need score!
    # print(faithfulness_df["score"].mean(), relevancy_df["score"].mean())


loop = asyncio.run(batch())




