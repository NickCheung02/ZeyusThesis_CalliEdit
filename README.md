# CalliEdit

**生成带有题字的中国山水画。** 输入画面描述、题字内容和位置，由 SD3 与两个 ControlNet 分支共同生成完整画面。

项目基于 [PosterMaker](https://github.com/alimama-creative/PosterMaker) 改造，当前保留场景生成与文字渲染能力，已移除商品图片输入、主体掩码与保留、Logo/小字特殊遮挡，以及 SAM 检测和奖励训练。

## 当前状态

| 功能 | 状态 |
| --- | --- |
| 无显卡、离线 CPU 冒烟测试 | 已通过，13 项测试 |
| 完整画布生成链路 | 已实现，小型模型验证通过 |
| 旧 PosterMaker 权重兼容 | 已实现，模拟旧权重验证通过 |
| 题字内容与位置控制 | 支持最多 7 个框，每框最多 16 字 |
| 竖排题字 | 可采用一字一框的方式表达 |
| 无题字生成 | 推理支持 `texts=[]` |
| 命令行推理、Gradio 界面 | 已提供；真实权重出图及界面运行尚未验证 |
| 山水画本地数据接入 | 已完成；题字数据用于 Stage 1，纯图像用于 Stage 2 |
| 正式 GPU 训练 | 尚未验证 |

冒烟测试验证代码能执行，不代表已获得可用的山水画或书法生成质量。

## 1. 快速开始：CPU 冒烟测试

在已安装项目测试依赖的 Python 环境中，从项目根目录运行：

```bash
python smoke_test.py
```

也可以从其他目录使用脚本的绝对路径执行，脚本会自动切换到项目根目录。

测试强制使用 **CPU、float32 和离线模式**，固定随机种子，使用临时缓存。无需显卡、预训练大模型、训练数据或 Gradio，也不依赖 DeepSpeed、Triton、xFormers。需要保留仓库中的字符特征文件：

```text
assets/char2feat_ppocr_neck64_avg.pth
```

测试使用的主要依赖为 PyTorch、torchvision、diffusers、transformers、NumPy、Pillow、OpenCV 和 safetensors，以及它们的运行依赖。当前电脑已在 PyTorch 2.3.0 环境下通过测试；仓库的 `requirements.txt` 是包含 GPU 训练组件的依赖清单，不是 CPU 冒烟测试专用安装清单。

成功时输出类似：

```text
CPU smoke test | PyTorch 2.3.0 | offline | float32
...
Ran 13 tests in ...s

OK
```

测试覆盖：

- 旧场景权重的 17→16 通道转换、新权重原样加载及层数识别。
- 文字权重的原版双字典格式和 Wrapper 格式。
- 参数缺失、形状不匹配等错误的严格检查。
- 空题字、竖排字框、非法坐标与输入不可变性。
- 双分支管线在有、无 CFG 引导时的去噪。
- 模拟旧权重加载、字符编码、两步去噪、VAE 解码、PNG 保存与文件校验。
- 冻结主干后的场景分支反向传播，以及梯度有限且非零。

图像网络使用随机初始化的小模型，场景描述使用预生成的小型文本特征。测试图片存放在临时目录，完成后自动清理。测试不覆盖完整 CLIP/T5 文本编码、真实预训练权重画质、Gradio 界面或多卡训练。

## 2. 模型结构

```text
山水画描述 → CLIP / T5 → 场景文本特征 ───────────┐
                                              │
题字内容与坐标 → 字符及位置编码 → Adapter        │
                                ↓             │
                         TextRenderNet         │
                                ↓             ↓
初始噪声 → SD3 去噪 ← 控制残差融合 ← SceneGenNet
               ↓
            VAE 解码
               ↓
           带题字的画作
```

| 组件 | 代码名称 | 职责 |
| --- | --- | --- |
| SD3 Transformer | `SD3Transformer2DModel` | 根据文本与控制残差进行去噪 |
| 场景分支 | `controlnet_scene` | 学习场景生成控制 |
| 文字分支 | `controlnet_text` | 根据字符与位置特征控制题字 |
| 文字适配器 | `LinearAdapterWithLayerNorm` | 将 128 维题字特征映射到模型条件维度 |
| 字符编码器 | `TextEmbedder` | 拼接 OCR 字符特征、字符位置编码和文字框位置编码 |
| VAE | `AutoencoderKL` | 图像与 latent 之间的编码、解码 |

两个 ControlNet 分支当前均使用全零的图像 latent 条件；训练与推理采用相同方式。没有参考商品图或需要保留的主体区域，整幅画面参与生成。

文字特征由 64 维 OCR 字符特征、32 维字符位置编码和 32 维文字框位置编码组成。题字位置会映射到编码器使用的 1024×1024 坐标系。

## 3. 项目结构

```text
.
├── app.py                         # Gradio 界面，首次生成时加载模型
├── inference.py                   # 命令行推理与共享模型加载入口
├── inference.sh                   # 推理示例
├── smoke_test.py                  # 离线 CPU 冒烟测试入口
├── inspect_dataset.py             # 本地数据索引、字框与图像检查
├── train_sd3_stage1.py            # 第一阶段：训练文字分支与 Adapter
├── train_sd3_stage2.py            # 第二阶段：训练场景分支
├── train.sh                       # 两阶段训练示例
├── models/                       # Transformer、ControlNet、文字编码及 OCR
├── pipelines/pipeline_sd3.py      # 双 ControlNet 生成管线
├── utils/
│   ├── checkpoint_utils.py       # 新旧权重兼容与严格加载
│   ├── data_processor.py         # 题字校验与推理输入准备
│   ├── sd3_utils.py              # 基础模型与文本特征工具
│   ├── eval_utils.py             # 验证与 OCR 相关逻辑
│   └── args_utils.py             # 训练参数
├── data_utils/                   # 训练、验证 Dataset
├── configs/                      # Accelerate / DeepSpeed 配置
├── tests/                        # CPU 回归与冒烟测试
├── assets/                       # 字符特征字典、字体
├── dataset/                      # 本地数据与制作脚本，Git 忽略
├── checkpoints/                  # 外部模型权重，需自行准备，Git 忽略
└── images/results/               # 生成结果，Git 忽略
```

## 4. 准备真实推理环境与权重

真实推理需要基础 SD3 模型和两个分支权重。默认文件组织如下：

```text
checkpoints/
├── stable-diffusion-3-medium-diffusers/
│   ├── transformer/
│   ├── vae/
│   ├── scheduler/
│   ├── text_encoder/             # 同时需要 text_encoder_2、text_encoder_3
│   └── tokenizer/                # 同时需要 tokenizer_2、tokenizer_3
└── our_weights/
    ├── scenegen_net.pth
    └── textrender_net.pth
```

原始模型来源：

- [Stable Diffusion 3 Medium](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers)
- [PosterMaker](https://huggingface.co/alimama-creative/PosterMaker)

推理代码在 CUDA 可用时使用 CUDA，否则使用 CPU。`--use_float16` 仅在 CUDA 可用时生效；CPU 使用 float32。完整 SD3 模型与 CPU 冒烟测试的小模型规模差异很大，当前尚未验证完整模型在这台电脑上的内存占用和运行时间。

`requirements.txt` 保留了原训练环境的依赖版本。正式 GPU 环境应根据实际 PyTorch/CUDA 配置检查依赖兼容性；当前 CPU 测试通过并不意味着该清单的全量安装已验证。

## 5. 复用旧 PosterMaker 权重

直接传入旧文件路径即可，不需要覆盖或手动修改原始权重。

| 权重 | 兼容行为 |
| --- | --- |
| SD3、VAE、CLIP/T5 | 按原基础模型目录加载 |
| 旧 SceneGenNet | 将 `pos_embed_input.proj.weight` 的最后一个掩码输入通道裁掉，17→16 |
| 新 SceneGenNet | 原样严格加载 |
| TextRenderNet + Adapter | 支持原版双字典格式及训练保存的 Wrapper 扁平 `state_dict` |

场景与文字分支的初始化层数分别从权重识别，优先于配置中的默认层数。其他参数缺失、多余或形状错误均报错，不使用 `strict=False` 跳过问题。

兼容加载器支持单文件 `.pth`、`.bin`、`.safetensors`。场景权重也支持包含 `diffusion_pytorch_model.safetensors` 或 `diffusion_pytorch_model.bin` 的本地 Diffusers 目录；暂不支持分片权重。文字检查点需同时包含文字分支与 Adapter。

已使用模拟旧权重验证：其余参数逐项保持不变，旧结构的掩码通道输入为零时，转换前后前向结果一致。新流程同时改变了图像条件，因此这不保证恢复旧海报管线的生成结果，也不保证未经微调就能生成理想的书法山水画。

## 6. 命令行生成

从项目根目录运行。以下示例直接复用旧权重文件名：

```bash
python inference.py \
  --pretrained_model_name_or_path=./checkpoints/stable-diffusion-3-medium-diffusers \
  --controlnet_model_name_or_path=./checkpoints/our_weights/scenegen_net-0415.pth \
  --controlnet_model_name_or_path2=./checkpoints/our_weights/textrender_net-0415.pth \
  --prompt='Chinese ink landscape painting, misty mountains, a river and pine trees, rice paper texture' \
  --texts='[{"content":"山","pos":[850,120,930,220]},{"content":"静","pos":[850,240,930,340]}]' \
  --resolution_w=1024 --resolution_h=1024 \
  --num_inference_steps=28 --cfg_scale=5.0 \
  --seed=42 --num_images_per_prompt=1 \
  --use_float16
```

也可将权重放到默认文件名对应的位置后执行 `bash inference.sh`。

| 参数 | 含义 | 默认值 |
| --- | --- | --- |
| `--pretrained_model_name_or_path` | 基础 SD3 目录 | 需指定 |
| `--controlnet_model_name_or_path` | 场景分支权重 | 需指定 |
| `--controlnet_model_name_or_path2` | 文字分支与 Adapter 权重 | 需指定 |
| `--prompt` | 画面描述 | 内置英文山水描述 |
| `--texts` | 题字框 JSON 数组 | `[]` |
| `--resolution_w` / `--resolution_h` | 输出画布宽高 | `1024` / `1024` |
| `--num_inference_steps` | 去噪步数 | `28` |
| `--cfg_scale` | 提示词引导强度 | `5.0` |
| `--seed` | 随机种子 | `0` |
| `--num_images_per_prompt` | 生成张数 | `1` |
| `--output_dir` | 保存目录 | `./images/results` |

输出文件名为 `landscape_<seed>_<index>.png`。相同种子和序号再次写入同一目录时会覆盖原结果。

### 题字输入规则

```json
[
  {"content": "山", "pos": [850, 120, 930, 220]},
  {"content": "静", "pos": [850, 240, 930, 340]}
]
```

- `pos` 为输出画布上的像素坐标 `[x1, y1, x2, y2]`，左上角为原点。
- 坐标必须有限、位于画布内，并满足 `x1 < x2`、`y1 < y2`。
- 画布宽高必须为正数且是 16 的倍数。
- 最多 7 个框，每框内容为 1–16 个字符；竖排可用一字一框。
- `[]` 表示不带题字，不需要构造空字符串文字框。

输入位置是模型的生成条件，不是生成后通过字体库叠字。当前未提供独立的书法字体或笔意控制接口。

## 7. Gradio 界面

安装 Gradio 等界面依赖并准备权重后运行：

```bash
python app.py
```

默认地址为 `http://localhost:7861`；服务绑定 `0.0.0.0:7861`。界面提供描述、题字 JSON、画布尺寸、随机种子、步数和引导强度。

默认权重路径为 `checkpoints/our_weights/scenegen_net.pth` 与 `textrender_net.pth`。也可以指定原始文件：

```bash
CALLIEDIT_SCENE_CHECKPOINT=./checkpoints/our_weights/scenegen_net-0415.pth \
CALLIEDIT_TEXT_CHECKPOINT=./checkpoints/our_weights/textrender_net-0415.pth \
python app.py
```

布局预览只画题字框与序号，序号对应 JSON 中的顺序。模型在首次点击生成时加载；当前界面尚未进行运行验证。

## 8. 两阶段训练

训练仍沿用两阶段结构：

| 阶段 | 训练参数 | 冻结参数 |
| --- | --- | --- |
| Stage 1 | TextRenderNet、Adapter | SD3、VAE、文本编码器、场景分支 |
| Stage 2 | SceneGenNet | SD3、VAE、文本编码器、文字分支与 Adapter |

训练示例：

```bash
bash train.sh 1
TEXT_CHECKPOINT=/path/to/stage1.pth bash train.sh 2
```

`train.sh` 是可编辑的参数示例，默认输出目录为 `/tmp`，从零初始化场景分支。若需要以旧场景权重初始化，应在相应训练命令中添加 `--controlnet_model_name_or_path=/path/to/scene.pth`。

`--resume_from_checkpoint` 恢复模型权重及脚本解析的步数信息，不恢复完整优化器状态。权重兼容入口同样适用于训练加载；该行为不等同于完整训练状态续跑。

### 本地数据接入

`data_utils/landscape_dataset.py` 统一处理两阶段的训练与验证数据；原 Dataset 模块保留为训练脚本的兼容入口。无需手动生成新的 JSON 或复制图片。

| 阶段 | 图像及标注来源 | 描述来源 | 训练 / 验证样本数 |
| --- | --- | --- | --- |
| Stage 1：写字 | `dataset1_manu_made_1` 至 `_4` 的 `images/` 和 `training_data.json` | JSON 的 `caption` | 9,088 / 480 |
| Stage 2：画图 | `dataset2_only_landscape/images/` | 同目录 XLSX 的“文字描述”单元格中的 Description 4 | 9,020 / 480 |

纯图像表格名为 `data1_9000data_nowords_COMBINED.xlsx`，使用 `image_name` 列与图片精确匹配。
描述会先还原字面量 `\n`；有编号时提取 `Description 4:` 的内容，无编号时取四段中的第 4 段。
无法解析的描述计入报告并排除，不回退到整段描述。未列入表格的图片不自动参与训练。
纯图像传入全零题字特征，不再被无文字过滤规则排除。

默认以文件名分组，按固定种子 42 划分约 5% 验证集，同一底图的不同字体版本不会分散到训练集与验证集。
可通过训练参数 `--dataset_root`、`--validation_fraction`、`--dataset_split_seed` 调整。
以上数量对应当前本地数据及默认划分，不是硬编码。

全量检查发现题字数据中有 180 个零面积等无效字框；加载时剔除这些标注，仍保留有有效题字的图片。
原图、JSON、Excel 均不改写。其余坐标与容量错误会明确报错，不用其他样本静默替换。
图片保持比例缩放后在右侧、底部补白；字框同步变换，训练与推理共用题字编码规则。

```bash
# 查看两个阶段的数量与划分
python inspect_dataset.py

# 逐一检查图片头、文件完整性与题字坐标
python inspect_dataset.py --check_images
```

当前全量检查：Stage 1 共 9,568 张，Stage 2 共 9,500 张，清理无效字框后图像/坐标检查错误为 0。
数据批量读取与无文字验证分支已接通；Stage 2 不执行空题字 OCR 评分。
`openpyxl` 是读取描述表的新增依赖，已列入 `requirements.txt`。
本地 `dataset/` 仍被 Git 忽略，数据兼容代码及测试均放在受版本管理的目录中。

当前电脑的 CPU 冒烟测试包含数据描述解析、两阶段批量读取、分组切分及小型反向传播检查，不代表完整训练脚本已在 CPU 或 GPU 环境完成验证。

## 9. 来源与许可

本项目基于 PosterMaker，原源码中的版权与许可声明予以保留。

- 论文：[PosterMaker: Towards High-Quality Product Poster Generation with Accurate Text Rendering](https://arxiv.org/abs/2504.06632)
- 原仓库：[alimama-creative/PosterMaker](https://github.com/alimama-creative/PosterMaker)

基础 SD3 模型和派生权重的使用需遵循对应模型的许可条款。本文档不另行授予基础模型或第三方权重的使用权。
