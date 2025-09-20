# 地形重建 — 完整产品设计

目标：本地科学重建比较工作台v1，内部terraintrace。本文件仍是完整用户体验与领域行为的规范，不因实现进度而删减需求。

## 1. 要回答的问题

在保持低分辨率观测一致的同时，重建方法如何权衡高程误差与坡度/边缘结构？
用户载入一个地形案例 → 同步查看真值/LR/三种重建 → 看误差与剖面 → 理解方法取舍 → 下载数组和实验配置。
这是科学计算工具，不是“AI把图变清晰”。首版不用深度网络、真实遥感服务或教师私有数据。

## 2. 已有与先修

TerrainCase含id/group/high_res/scale/pixel_size，合成LR由HR面积均值生成。nearest、投影bilinear、train-only径向spectral_prior可复用；运行必须保存真实数组并由前端直接读取，不能由汇总指标伪造热图。
默认12个48×48合成场，train/val/test各4、组互斥。seed17频谱先验仅小幅改善RMSE而坡度/边缘更差，产品主线必须保留取舍。
指标协议固定为 terrain_v2：gradientCorrelation 对二维 gx/gy 拼接计算，零梯度方差返回 null 和原因；旧 axis0 指标保留旧名称。
LR-only 推断使用独立 DTO，不能为运行旧 TerrainCase 偷偷构造假 HR。

## 3. 输入、方法与指标

v1输入两类：
1. benchmark：NPZ的high_res数组 + JSON案例metadata（id/group/split/scale/pixel_size/elevation_unit）。由显式area_mean生成LR，标“受控降采样评估”。
2. inference：只有low_res + scale/pixel_size/elevation_unit。没有HR就没有RMSE/MAE/坡度误差；v1 不接受任意 prior_id。
JSON不能嵌大数组，上传NPZ使用allow_pickle=false，先检查ZIP未压缩体积≤32MiB/文件。数组二维、有限、每维16–512（HR），scale限2/4/8且尺寸整除；LR输出HR不得超512。每数据集≤60案例，等方形像素；NoData/非有限值首版拒绝并定位，不自动补零。
合成数据单位写“合成高程单位”；用户数据必须指定单位和像素间距。NPZ没有地理范围时只承诺按声明group隔离，不声称验证地理不重叠。GeoTIFF/CRS/垂直基准/NoData转换留作下一版本，不是本版隐含必做项。

组不得跨split，精确重复数组hash不能跨split；先验只看train，validation做方法选择，test用于冻结协议后的报告。仅LR可用nearest/bilinear，spectral要求已有prior的scale、HR形状、pixel_size、单位协议匹配。
指标：高程RMSE/MAE、二维坡度MAE、二维梯度相关、Laplacian边缘RMSE、LR一致性RMSE/最大漂移。描述单位及方向；投影一致性的数值容差使用相对1e-8，不把浮点误差包装成物理精度。
多seed的mean/pstdev是稳定性描述，不是置信区间。ROI指标是探索，不替代整个test结果。

## 4. 页面与视觉

灰白 #eef3f2、深墨 #263a3b、青绿 #287e79。高程统一顺序色标，误差使用零中心发散色标；色条与数值范围必须固定在图旁。
1440主区三幅并排可联动热图，上方数据/案例选择，下方横向剖面和指标表；左侧案例列表200px可收起。1100两图+第三tab，390单图切换且记住视窗。
不以三张各自自动拉伸色标的“漂亮图”暗示改善。

### 数据 /datasets/:id
样例库按train/val/test组，上传NPZ和metadata后预览shape/单位/范围/组/异常；缺metadata提示具体字段。可选benchmark/inference，决定后续显示能力。
运行配置：方法与指标协议；scale、pixel_size、单位都固定在数据集 revision。演示合成数据的 seed 属于数据集创建配置；当前重建方法是确定性的，不提供不会改变计算的 per-run seed。频谱先验只从当前 benchmark revision 的 train 案例拟合；inference 不支持频谱先验。

### 比较 /runs/:id/cases/:case
面板默认真值、bilinear、spectral；可切nearest、LR、绝对/有符号误差。同步平移/缩放、十字光标和坐标读数；切方法保留同一区域。
有HR时同一案例所有高程图共用包含真值与重建的色标；误差共用±最大绝对误差。仅LR时高程范围依据LR与全部显示重建，误差选项隐藏。
拖动剖面端点：默认128个采样，双线性取样，服务端计算同一坐标路径。曲线显示距离/高程，图上端点可键盘微调；下载相同采样CSV。
ROI框选只改变探索表，旁边保留全案例指标；明确“局部探索”标识。原点在左上，row向下、col向右；空间单位由pixel_size换算，避免剖面方向不一致。

### 总览与先验 /runs/:id
按方法/案例展示指标矩阵，可排序但不合成一个神秘总分。选择RMSE改善而edge变差的案例，看图与剖面解释。
先验详情面板显示训练组数组hash、径向增益曲线、bins与clip范围、协议及兼容输入；test不能重新拟合。无HR输入仅展示观测一致性，不出现虚构真值分数。

## 5. API与产物

- POST /api/v1/terrain-datasets：multipart NPZ+metadata → id/revision、shape/单位/分组检查。`mode=inference` 只接受一个 low_res 文件。
- GET /api/v1/terrain-datasets/{id}/{revision}：读取固定数据清单。
- POST /api/v1/reconstruction-runs：dataset_id/revision、mode、methods、metrics_protocol → 202 job；scale 和输入单位从 revision 读取。
- GET /api/v1/jobs/{id}、GET /api/v1/jobs/{id}/events、POST /api/v1/jobs/{id}/cancel：持久化进度、SSE 续读和案例边界取消。
- GET /api/v1/runs/{id}/cases：案例/组/split/状态/摘要；GET /api/v1/runs/{id}/cases/{case_id}/metrics：案例级协议与指标，无真值的字段为 null。
- GET /api/v1/runs/{id}/cases/{case_id}/arrays?method=&kind=：原数组用 binary NPY；kind 支持 reconstruction、low_res、high_res、difference 和 absolute_error。元信息通过 headers 提供 shape/dtype/SHA-256/value range；`display=true` 提供≤256×256展示网格。
- POST /api/v1/runs/{id}/profiles：case_id、methods、points[[row,col],[row,col]]、samples=128 → 服务端共用距离、方法曲线及 benchmark 真值曲线。
- POST /api/v1/runs/{id}/roi：整数 bounds → 局部统计，标 exploratory。
- POST /api/v1/runs/{id}/exports?case_id=：登记由真实输入、数组、指标、profile CSV、静态 HTML、配置和 manifest 组成的 ZIP；GET /api/v1/artifacts/{id}/download 下载。
- 频谱先验参数与训练案例hash放在 run 的 prior metadata 中；v1 不暴露任意 prior_id 或独立 GET /priors/{id}。

新增重建数组由methods真实输出并登记，不从指标倒推。绘图数据降采样不改变科学指标；前端不能把256×256显示网格重新当原数组计算分数。
配置例：
~~~json
{"dataset_id":"terrain_1","dataset_revision":1,"mode":"benchmark","methods":["nearest","bilinear","spectral_prior"],"metrics_protocol":"terrain_v2"}
~~~

## 6. 实施与完成

A. 二维梯度指标、独立inference DTO、数组产物与兼容prior校验。
B. NPZ/metadata导入、group/hash隔离、API和案例级取消/重跑。
C. 联动热图、统一色标、剖面/ROI、全案例指标与无HR状态。
D. 数组/CSV/HTML导出、可复现教程和一组明确的误差/结构取舍案例。

必须实际验证：同坐标多图一致，切方法保留视窗；剖面CSV能重画原曲线；下载重建数组能重算面积一致性；无HR不给真值指标；相同数组不能跨train/test；二维gradient协议区别于旧axis0。
预览图只展示设计样例，完成时必须换成真实重建数组，不保留程序化假热图。
