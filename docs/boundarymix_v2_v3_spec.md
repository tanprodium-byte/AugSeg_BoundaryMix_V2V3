Tôi đang làm nghiên cứu BoundaryMix cho Semi-Supervised Semantic Segmentation, dựa trên AugSeg.

Repo hiện tại:

https://github.com/tanprodium-byte/AugSeg_BoundaryMix

Bối cảnh:

- Base framework: AugSeg.
- Dataset chính: PASCAL VOC semi-supervised 662 labels.
- Backbone: ResNet-101.
- Crop: 513.
- Batch: 4/GPU × 2 GPU = global batch 8.
- Hiện tại đã có BoundaryMix V1 chạy được với các bản A0/A1/A2/A3/A4/A5.
- A0 = AugSeg baseline.
- A1 = neutral control, bật BoundaryMix branch nhưng gamma_in = gamma_out = 1, không confidence.
- A4 = V1 inner/outer fixed reweight, gamma_in = 0.7, gamma_out = 0.3.
- A5 = A4 + confidence modulation.

Kết quả V1 hiện tại cho thấy:

- Các bản V1 không phá baseline.
- A4 có Best EMA cao nhất sơ bộ.
- Nhưng A1 neutral control cũng rất mạnh, nên chưa thể claim fixed boundary weighting là nguyên nhân chính.
- A5 thấp hơn A4, gợi ý pixel-level confidence modulation thô chưa tốt.
- Vì vậy hướng mới không dùng fixed gamma 0.7/0.3 làm main nữa.

Trục nghiên cứu chính:

Artificial CutMix seam/boundary reliability trong Semi-Supervised Semantic Segmentation.

Không đề xuất class-aware CutMix, không đi theo AEL/DeS4, không thêm trick rời rạc. Tập trung vào CutMix boundary, component bị boundary cắt/che, và quan hệ local giữa hai phía boundary.

=====================================================================
MỤC TIÊU
=====================================================================

Hãy giúp tôi cài đặt 3 phiên bản mới, tách biệt rõ ràng để ablation sạch:

1. V2 — Boundary-Affected Component Weighting only
2. V3 — JS Boundary Compatibility only
3. V2 + V3 — Component Weighting + JS Boundary Compatibility

Mỗi phiên bản phải có config riêng, bật/tắt độc lập được.

Ràng buộc quan trọng:

- Không được để V3 ngầm chứa V2.
- Không được để V2 ngầm chứa V3.
- Khi boundary_component.enabled=false và boundary_compatibility.enabled=false, code phải chạy như A1 hoặc baseline tương ứng tùy config.
- V3 standalone không dùng component gate q_C, tức đặt q_C = 1.
- V2+V3 mới dùng q_C của V2 để gate thêm cho BCR.

=====================================================================
NGUYÊN TẮC CÀI ĐẶT CHUNG
=====================================================================

Trước khi sửa code:

1. Đọc kỹ repo.
2. Xác định file train chính, chỗ tạo CutMix/adaptive CutMix, chỗ có mask M, chỗ tạo mixed image/mixed label, chỗ tính unsupervised/mixed loss.
3. Không sửa lung tung pipeline AugSeg.
4. Mọi thay đổi phải có config bật/tắt.
5. Khi tất cả module mới disabled, kết quả/loss path phải giống baseline hiện tại.
6. Các gate, teacher probabilities, pseudo-labels, component weights nên detach / no_grad nếu chúng chỉ dùng để tạo weight/gate, tránh module tự tối ưu weight để né loss.
7. Tránh NaN: clamp xác suất với epsilon khi tính JS/KL.
8. Log đầy đủ các thống kê để debug.
9. Giữ code backward-compatible với A0/A1/A4.

=====================================================================
PHIÊN BẢN 1: V2 — BOUNDARY-AFFECTED COMPONENT WEIGHTING ONLY
=====================================================================

Ý tưởng:

V1 chỉ xử lý vài pixel boundary:

B_in = M - erode(M)
B_out = dilate(M) - M

rồi nhân gamma cố định.

V2 không làm vậy. V2 dùng CutMix mask M để tìm semantic component nào bị boundary cắt qua.

Nói ngắn:

CutMix mask M
→ connected components trên pseudo-label target
→ tìm component C bị M cắt qua
→ tính visible ratio, visible area, mean confidence
→ tạo soft component weight q_C
→ dùng q_C để weight mixed CE

---------------------------------------------------------------------
Quy ước mask
---------------------------------------------------------------------

M(i) = 1 nghĩa là pixel i lấy từ source side.
M(i) = 0 nghĩa là pixel i lấy từ target side.

Trong AugSeg hiện tại:

source side A = labeled image hoặc mixed source
target side B = unlabeled image với pseudo-label teacher

Mixed image:

x_mix = M ⊙ x_A + (1 - M) ⊙ x_B

Mixed label:

y_mix = M ⊙ y_A + (1 - M) ⊙ y_B

Với target unlabeled:

y_B = y_hat_u

là pseudo-label từ teacher.

---------------------------------------------------------------------
Connected component kiểu B
---------------------------------------------------------------------

Chọn kiểu B:

Chạy connected component trên toàn pseudo-label/label map trước, rồi lọc component nào bị CutMix mask chia qua hai phía.

Dùng 8-connectivity.

Với một component C, component đó bị CutMix boundary ảnh hưởng nếu:

C ∩ M != empty

và:

C ∩ (1 - M) != empty

Tức là component có pixel nằm cả trong và ngoài mask, nghĩa là boundary đã cắt qua nó.

---------------------------------------------------------------------
Target component
---------------------------------------------------------------------

Với target pseudo component C_B:

Phần còn visible sau CutMix:

C_B^vis = C_B ∩ (1 - M)

Phần bị che bởi source patch:

C_B^occ = C_B ∩ M

Visible ratio:

v_B(C) = |C_B^vis| / |C_B|

Mean confidence trên phần visible:

R_C = (1 / |C_B^vis|) * sum_{i in C_B^vis} c_u(i)

Trong đó:

c_u(i) = max_k p_u^t(i,k)

---------------------------------------------------------------------
Visible score
---------------------------------------------------------------------

g(v) = clip((v - tau_low) / (tau_high - tau_low), 0, 1)

Gợi ý default:

tau_visible_low: 0.2
tau_visible_high: 0.6

Ý nghĩa:

v <= 0.2  → component còn quá ít, score gần 0
v = 0.4   → score khoảng 0.5
v >= 0.6  → component còn đủ nhiều, score 1

---------------------------------------------------------------------
Area score
---------------------------------------------------------------------

Gọi:

a = |C_B^vis|

Area score:

h(a) = clip((a - a_min) / (a_max - a_min), 0, 1)

Gợi ý default:

area_min: 64
area_max: 512

Có thể cần chỉnh theo resolution feature/output thực tế.

---------------------------------------------------------------------
Soft component weight
---------------------------------------------------------------------

q_C = R_C * g(v_B(C)) * h(|C_B^vis|)

Ý nghĩa:

teacher confident
+ component còn visible đủ nhiều
+ fragment đủ lớn
→ q_C cao

teacher không chắc
hoặc component bị che quá nhiều
hoặc fragment quá nhỏ
→ q_C thấp

---------------------------------------------------------------------
Pixel weight cho V2
---------------------------------------------------------------------

Tôi muốn V2 dùng soft component weight, không hard ignore.

Với pixel i thuộc phần visible của component bị boundary ảnh hưởng:

w_i = base_i * q_C

Trong đó base_i nên configurable:

base_pixel_weight: "one"        # default để gần A1 hơn
# hoặc
base_pixel_weight: "confidence" # dùng c_i nếu muốn ablation

Default tôi muốn ưu tiên:

base_i = 1

để tránh lặp lại vấn đề A5 confidence modulation quá aggressive.

Nhưng R_C vẫn được dùng trong q_C, tức confidence xuất hiện ở cấp component, không nhân thô từng pixel.

Với target pixel không thuộc component bị boundary cắt:

w_i = base_i

Với source labeled GT pixel:

w_i = 1

Loss V2:

L_mix^V2 = sum_i w_i CE(p_s(i), y_mix(i)) / (sum_i w_i + eps)

---------------------------------------------------------------------
Config V2 đề xuất
---------------------------------------------------------------------

boundary_component:
  enabled: true

  connectivity: 8
  apply_to: "target_only"
  foreground_only: true

  tau_visible_low: 0.2
  tau_visible_high: 0.6

  area_min: 64
  area_max: 512

  use_mean_confidence_in_q: true
  base_pixel_weight: "one"     # "one" hoặc "confidence"
  weight_mode: "soft"

  debug_log: true

---------------------------------------------------------------------
Log cần có cho V2
---------------------------------------------------------------------

Mỗi epoch hoặc mỗi N iteration, log:

num_affected_components
num_affected_pixels
visible_ratio_mean/std/min/max
visible_area_mean/std/min/max
mean_confidence_R_C
q_C_mean/std/min/max
weighted_loss_denominator
per-class affected component count
per-class mean q_C

Visualization nếu có thể:

CutMix mask M
pseudo-label target
affected components
q_C heatmap / weight map

=====================================================================
PHIÊN BẢN 2: V3 — JS BOUNDARY COMPATIBILITY ONLY
=====================================================================

Ý tưởng:

V3 không xử lý component weighting. V3 chỉ học quan hệ local giữa hai phía CutMix boundary.

V3 trả lời câu hỏi:

Hai phía sát CutMix boundary nên liên tục hay phân tách về feature?

Ví dụ:

road | road       → feature nên liên tục hơn
car  | road       → feature không nên quá giống
uncertain pair    → bỏ qua

V3 standalone không dùng q_C. Đặt:

q_C = 1

cho mọi component.

---------------------------------------------------------------------
Boundary bands
---------------------------------------------------------------------

Từ CutMix mask M, tạo hai dải boundary:

S_in^b = M - erode_b(M)

S_out^b = dilate_b(M) - M

Trong đó:

S_in^b  = dải phía trong/source side gần boundary
S_out^b = dải phía ngoài/target side gần boundary

band_width = b là tham số cần tuning.

---------------------------------------------------------------------
Local mask-aware cross-boundary pairs
---------------------------------------------------------------------

Không dùng patch vuông làm main. Dùng local band-to-band pairs.

Tạo tập pairs:

P = {(a,b) | a in S_in^b, b in S_out^b, ||a-b|| <= d}

Trong đó:

- a: vị trí/cell phía trong mask.
- b: vị trí/cell phía ngoài mask.
- d: radius local relation.

Tôi muốn ablation:

d = 1
d = 2
d = 3

để biết local relation tốt nhất ở phạm vi nào.

Lưu ý:

- Nếu tính trên ảnh gốc, d=1 gần như pixel-pair.
- Nếu tính trên feature map stride 8/16, một cell đã có receptive field lớn.
- Dù vậy, d=1 chỉ nên xem là diagnostic; d=2 hoặc d=3 có thể hợp lý hơn.
- Cần chạy ablation d=1,2,3.

Pair generation có thể dùng:

pair_mode: "radius"
# hoặc
pair_mode: "topk"
topk: 3 hoặc 5

Nên giới hạn số pair mỗi image:

max_pairs_per_image: 2048

để không nổ chi phí.

---------------------------------------------------------------------
Semantic distribution cho mỗi pair
---------------------------------------------------------------------

Với mỗi pair (a,b):

Nếu phía đó là GT:

p_a = onehot(y_a)

Nếu phía đó là pseudo side:

p_a = p_a^t

Tương tự cho p_b.

Tức p_a,p_b là semantic probability distributions hai phía boundary.

---------------------------------------------------------------------
JS semantic compatibility
---------------------------------------------------------------------

Với mỗi pair (a,b):

m_ab = 0.5 * (p_a + p_b)

JS(p_a,p_b) =
0.5 * KL(p_a || m_ab) + 0.5 * KL(p_b || m_ab)

Dùng natural log, normalize:

JS_norm(p_a,p_b) = JS(p_a,p_b) / log(2)

Semantic compatibility:

s_sem^{ab} = 1 - JS_norm(p_a,p_b)

Ý nghĩa:

s_sem gần 1:
    hai phía boundary có semantic distribution giống nhau

s_sem gần 0:
    hai phía khác semantic rõ

Cần clamp:

p = clamp(p, eps, 1)
m = clamp(m, eps, 1)

để tránh NaN.

---------------------------------------------------------------------
Feature compatibility
---------------------------------------------------------------------

Lấy feature student tại vị trí a,b:

F_a, F_b

Normalize:

F_a_norm = F_a / ||F_a||_2
F_b_norm = F_b / ||F_b||_2

Cosine similarity về [0,1]:

s_F^{ab} = (1 + F_a_norm^T F_b_norm) / 2

Cần xác định feature layer dùng cho F:

feature_layer: "decoder" hoặc "encoder_high" hoặc layer hiện repo dễ lấy nhất

Ban đầu ưu tiên layer gần segmentation head để align semantic.

---------------------------------------------------------------------
Gate trong V3 standalone
---------------------------------------------------------------------

V3 vẫn có gate, nhưng không có component gate q_C.

Gate standalone:

r_ab = R_a * R_b

Trong đó:

- Nếu vị trí thuộc GT side:

R = 1

- Nếu vị trí thuộc pseudo side:

R = c(i)

Có thể dùng soft gate:

r_ab = R_a * R_b

hoặc hard confidence threshold. Default dùng soft gate.

---------------------------------------------------------------------
Tri-state JS Boundary Compatibility Loss
---------------------------------------------------------------------

Đặt:

tau_same: 0.8
tau_diff: 0.3
margin: 0.4

Same-semantic pair:

Nếu:

s_sem^{ab} > tau_same

thì:

L_same^{ab} = r_ab * (1 - s_F^{ab})^2

Ý nghĩa:

hai phía semantic giống nhau
→ feature nên liên tục

Different-semantic pair:

Nếu:

s_sem^{ab} < tau_diff

thì:

L_diff^{ab} = r_ab * max(0, s_F^{ab} - margin)^2

Ý nghĩa:

hai phía semantic khác rõ
→ feature không nên quá giống

Uncertain pair:

Nếu:

tau_diff <= s_sem^{ab} <= tau_same

thì bỏ qua.

Loss V3:

L_BCR =
[
  sum_{same pairs} r_ab(1-s_F^{ab})^2
  +
  sum_{diff pairs} r_ab max(0,s_F^{ab}-margin)^2
]
/
[
  sum_{same/diff pairs} r_ab + eps
]

Tổng loss V3 standalone:

L = L_sup + lambda_u L_mix^neutral + lambda_bcr L_BCR

Trong đó L_mix^neutral phải giống A1/neutral path, không dùng V2 component weighting.

---------------------------------------------------------------------
Config V3 đề xuất
---------------------------------------------------------------------

boundary_component:
  enabled: false

boundary_compatibility:
  enabled: true

  semantic_metric: "js"
  band_width: 3

  pair_mode: "radius"       # hoặc "topk"
  pair_radius: 1            # chạy ablation 1, 2, 3
  topk: 5
  max_pairs_per_image: 2048

  tau_same: 0.8
  tau_diff: 0.3
  margin: 0.4

  lambda_bcr: 0.01

  use_confidence_gate: true
  use_component_gate: false

  feature_layer: "decoder"
  detach_teacher_distribution: true
  detach_gate: true

  debug_log: true

---------------------------------------------------------------------
Ablation V3 theo d
---------------------------------------------------------------------

Cần tạo ít nhất 3 config:

V3-d1:
    pair_radius = 1

V3-d2:
    pair_radius = 2

V3-d3:
    pair_radius = 3

Giữ cố định:

band_width = 3
tau_same = 0.8
tau_diff = 0.3
margin = 0.4
lambda_bcr = 0.01
max_pairs_per_image = 2048
use_component_gate = false

Mục tiêu:

Xem local cross-boundary relation nên học ở phạm vi sát seam hay cần context rộng hơn.

---------------------------------------------------------------------
Log cần có cho V3
---------------------------------------------------------------------

Log:

num_pairs_per_image
same_pairs_ratio
diff_pairs_ratio
uncertain_pairs_ratio
mean_s_sem
mean_s_F
mean_r_ab
mean_JS
L_BCR
pair_radius
band_width

Nên log theo d:

V3-d1 / V3-d2 / V3-d3

để so sánh.

=====================================================================
PHIÊN BẢN 3: V2 + V3 COMBINED
=====================================================================

Ý tưởng:

Bản combined dùng cả hai:

- V2 tạo soft component weight q_C.
- V3 dùng q_C để gate thêm cho boundary relation.

Pipeline:

CutMix mask M

V2:
→ tìm component bị M cắt
→ tính q_C
→ dùng q_C cho weighted mixed CE

V3:
→ tạo local cross-boundary pairs
→ tính JS semantic compatibility
→ tính cosine feature compatibility
→ gate bằng R_a R_b q_Ca q_Cb
→ tính L_BCR

---------------------------------------------------------------------
Gate combined
---------------------------------------------------------------------

Trong V2+V3:

r_ab = R_a * R_b * q_{C_a} * q_{C_b}

Nếu pixel không thuộc component bị boundary ảnh hưởng:

q_C = 1

Nếu component bị cắt quá nặng:

q_C ≈ 0

thì relation quanh nó không tác động mạnh.

---------------------------------------------------------------------
Loss combined
---------------------------------------------------------------------

L = L_sup + lambda_u L_mix^V2 + lambda_bcr L_BCR^V3

Trong đó:

L_mix^V2 =
sum_i w_i CE(p_s(i), y_mix(i)) / (sum_i w_i + eps)

và L_BCR dùng công thức JS tri-state như V3, nhưng gate là:

r_ab = R_a R_b q_Ca q_Cb

---------------------------------------------------------------------
Config combined đề xuất
---------------------------------------------------------------------

boundary_component:
  enabled: true

  connectivity: 8
  apply_to: "target_only"
  foreground_only: true

  tau_visible_low: 0.2
  tau_visible_high: 0.6

  area_min: 64
  area_max: 512

  use_mean_confidence_in_q: true
  base_pixel_weight: "one"
  weight_mode: "soft"

boundary_compatibility:
  enabled: true

  semantic_metric: "js"
  band_width: 3

  pair_mode: "radius"
  pair_radius: 2        # chọn d tốt nhất từ V3-d1/d2/d3
  topk: 5
  max_pairs_per_image: 2048

  tau_same: 0.8
  tau_diff: 0.3
  margin: 0.4

  lambda_bcr: 0.01

  use_confidence_gate: true
  use_component_gate: true

  feature_layer: "decoder"
  detach_teacher_distribution: true
  detach_gate: true

  debug_log: true

=====================================================================
BẢNG EXPERIMENT CẦN CHẠY
=====================================================================

Tối thiểu cần các version:

| Version | Mục tiêu |
|---|---|
| A0 | AugSeg baseline |
| A1 | neutral control |
| A4 | V1 fixed inner/outer weighting tốt nhất hiện tại |
| V2 | component soft weighting only |
| V3-d1 | JS boundary compatibility, d=1 |
| V3-d2 | JS boundary compatibility, d=2 |
| V3-d3 | JS boundary compatibility, d=3 |
| V2+V3-best | kết hợp V2 với V3 có d tốt nhất |

Nếu tài nguyên hạn chế:

Must run:
A0, A1, A4, V2, V3-d1, V3-d2, V2+V3-best

Optional:
V3-d3

Nhưng nếu muốn hiểu rõ d, nên chạy đủ d=1,2,3.

=====================================================================
CÁCH DIỄN GIẢI KẾT QUẢ
=====================================================================

Nếu V2 > A1:

Có bằng chứng rằng:

Boundary-affected component visibility/reliability có ích hơn neutral loss path.

Nếu V3-d* > A1:

Có bằng chứng rằng:

JS-based local boundary compatibility có ích.

Nếu V2+V3 > V2 và > V3:

Câu chuyện rất mạnh:

Component reliability và local boundary compatibility bổ sung nhau.

Nếu V2+V3 ≈ V2:

V2 là đóng góp chính, V3 chưa thêm nhiều.

Nếu V2+V3 ≈ V3:

Boundary relation là đóng góp chính, component weighting chưa rõ.

Nếu V2+V3 < V2 hoặc < V3:

Có thể:

lambda_bcr quá lớn
V2 weight làm gate quá mạnh
BCR và CE conflict
pair_radius quá rộng
JS threshold chưa phù hợp
feature layer chưa phù hợp

Trước tiên giảm:

lambda_bcr

hoặc thử d nhỏ hơn.

=====================================================================
THỨ TỰ TUNING
=====================================================================

Không tuning tất cả cùng lúc.

---------------------------------------------------------------------
Bước 1 — Sanity check
---------------------------------------------------------------------

Kiểm tra:

boundary_component.enabled=false
boundary_compatibility.enabled=false
→ loss giống A1/neutral path

V2 q_C forced = 1
→ loss giống A1

V3 lambda_bcr = 0
→ loss giống A1

V2+V3 use_component_gate=false
→ giống V3 standalone

---------------------------------------------------------------------
Bước 2 — Tune V2
---------------------------------------------------------------------

Giữ V3 off.

Thử:

tau_visible_low/high:
  (0.1, 0.5)
  (0.2, 0.6)
  (0.3, 0.7)

area_min/max:
  (32, 256)
  (64, 512)
  (128, 1024)

base_pixel_weight:
  one
  confidence

Ưu tiên default:

tau_visible_low = 0.2
tau_visible_high = 0.6
area_min = 64
area_max = 512
base_pixel_weight = one

---------------------------------------------------------------------
Bước 3 — Tune V3 standalone
---------------------------------------------------------------------

Giữ V2 off.

Thử:

pair_radius d = 1, 2, 3
lambda_bcr = 0.005, 0.01, 0.05
margin = 0.4, 0.5
tau_same/tau_diff = (0.8, 0.3), (0.85, 0.25)

Không chạy quá nhiều ban đầu. Ưu tiên:

d = 1, 2, 3
lambda_bcr = 0.01
tau_same = 0.8
tau_diff = 0.3
margin = 0.4

---------------------------------------------------------------------
Bước 4 — Combined
---------------------------------------------------------------------

Chọn best V2 và best V3-d.

Chạy:

V2 + V3-best

Nếu combined không ổn:

giảm lambda_bcr
giảm pair_radius
tắt component gate trong BCR để kiểm tra

=====================================================================
NHỮNG LỖI DỄ GẶP CẦN KIỂM TRA
=====================================================================

1. Mask M bị lệch resolution với output/feature.
2. Erode/dilate trên mask float sai, nên dùng binary mask.
3. Connected component chạy trên GPU khó, có thể làm CPU/NumPy/OpenCV trước để đúng logic.
4. Background VOC quá lớn, không nên component-filter background.
5. C^vis rỗng gây chia 0.
6. q_C collapse quá thấp làm denominator loss nhỏ.
7. JS NaN vì log 0.
8. Pair count quá lớn làm chậm.
9. V3 lấy pair quá xa làm loãng boundary.
10. BCR loss quá lớn so với CE.
11. Feature map stride không align với mask/pseudo-label.
12. Teacher distribution phải detach.
13. Gate phải detach.
14. Khi disabled module, result phải không đổi.

=====================================================================
IMPLEMENTATION SUGGESTION
=====================================================================

Nên tạo module riêng, ví dụ:

utils/boundary_component.py
utils/boundary_compatibility.py

Hoặc trong thư mục phù hợp với repo.

Các hàm nên có:

compute_component_weights(
    pseudo_label,
    confidence,
    cutmix_mask,
    num_classes,
    ignore_index,
    cfg
) -> weight_map, component_stats

compute_js_boundary_compatibility_loss(
    features,
    teacher_probs_or_mixed_probs,
    mixed_labels,
    cutmix_mask,
    confidence_map,
    component_weight_map_or_none,
    cfg
) -> loss_bcr, bcr_stats

Trong V3 standalone, component_weight_map_or_none=None, và khi đó q_C=1.

Trong V2+V3, truyền component weight map từ V2 vào BCR gate.

=====================================================================
KỲ VỌNG BÁO CÁO
=====================================================================

Không claim quá sớm.

Câu chuyện khoa học:

V1 fixed boundary weighting chỉ là diagnostic.
V2 kiểm tra boundary ảnh hưởng đến semantic component bị cắt/che.
V3 kiểm tra quan hệ local giữa hai phía artificial CutMix boundary.
V2+V3 kiểm tra hai tín hiệu này có bổ sung nhau không.

Main claim chỉ nên đưa sau khi có kết quả:

Gain phải vượt A0.
Quan trọng hơn, gain phải vượt A1 neutral control.

A1 là đối chứng rất quan trọng vì V1 cho thấy neutral branch cũng mạnh.

=====================================================================
DELIVERABLE TÔI MUỐN
=====================================================================

Khi làm việc trên repo, hãy:

1. Đọc code và xác định chính xác chỗ cần sửa.
2. Đề xuất plan cài đặt trước khi sửa.
3. Sau đó sửa code có kiểm soát.
4. Thêm config cho:
   - V2 only
   - V3-d1
   - V3-d2
   - V3-d3
   - V2+V3-best template
5. Thêm logging stats.
6. Thêm sanity check script hoặc hướng dẫn test nhỏ.
7. Không làm thêm class-aware CutMix.
8. Không thêm method ngoài trục BoundaryMix.
9. Không claim kết quả khi chưa train.
10. Giữ code backward-compatible với A0/A1/A4.

=====================================================================
TÓM TẮT NGẮN NHẤT
=====================================================================

V2:
component bị CutMix boundary cắt
→ q_C theo visible ratio, area, confidence
→ soft weighted CE.

V3:
local band-to-band cross-boundary pairs
→ JS semantic compatibility
→ cosine feature compatibility
→ same/diff/uncertain BCR loss.

V2+V3:
dùng q_C của V2 vừa weight CE vừa gate BCR.
