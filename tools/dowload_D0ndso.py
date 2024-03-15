from c2net.context import prepare, upload_output
c2net_context = prepare()

# 获取代码路径，数据集路径，预训练模型路径，输出路径
code_path = c2net_context.code_path
dataset_path = c2net_context.dataset_path
pretrain_model_path = c2net_context.pretrain_model_path
you_should_save_here = c2net_context.output_path


# 回传结果，只有训练任务才能回传
upload_output()