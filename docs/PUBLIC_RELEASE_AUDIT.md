# 公開前預查（2026-10-31）

使用者 2026-10-30 說：專題完成後 repo 要在 GitHub 公開，在那之前維持 private。專案規則不改寫 history，所以 history 裡的東西公開時都會看得到（除非屆時改用其他方式公開）。這份預查是唯讀的，提早列出 HEAD 與完整 history（278 個 commit）中需要處理或明確接受的項目。**沒有修改任何檔案。** 公開本身要在當時再取得使用者同意，公開前還要再查一次之後新增的 commit。

方法：4 個方向各由一個 agent 掃描，再由另一個 agent 獨立重做檢查並補漏。依 MACHINE_POLICY，**沒有搜尋密碼、金鑰或其他 credential**；公開前請使用者自己用 secret scanner（例如 gitleaks）掃完整 history。

## 沒有問題的部分

- **dataset 題目原文**：HEAD 與 history 都沒有。從 WikiText、HellaSwag、MMLU、ARC、WinoGrande 的本機檔各取多個句子，以 `git log --all -S` 搜尋，結果都是 0 個 commit。commit 進去的只有 005 允許的 index 清單、sha256 與計數。每題的 0 / 1 正確性向量與每段的 NLL 屬於衍生結果，不含原文。
- **serving 的 dataset**：都是 RandomDataset 的隨機 token，或只由 ' the' 組成的 prompt。
- **模型檔**：沒有任何 tokenizer、config、chat template 或模型卡。
- **不該進 git 的東西**：`.gitignore` 涵蓋的東西（raw/、models/、data/、*.gguf、*.kld、.env、*.pem、*.key 等）從來沒有被 commit。
- **大檔**：history 中最大的 blob 約 0.7 MB，pack 總共 7.7 MiB。
- **刪除的檔案**：history 中沒有任何刪除。只有一個被改名的舊 profile（C1 被取代的那一份），不敏感。
- **可定位的識別資訊**：沒有 IP（含 Tailscale 100.64/10 與私有網段）、MAC 位址、Mac 序號、Hardware UUID、GPU UUID，也沒有主機名稱或 Windows 使用者名稱。env capture 只記硬體型號層級的資訊，這是 benchmark 需要的。
- **他人資訊**：沒有其他人的名字或 email。

## 需要處理或明確接受的項目（都在 history 裡，不改寫 history 就無法移除）

| # | 項目 | 範圍 | 處理選項 |
|---|---|---|---|
| 1 | commit 的 author / committer email 是個人 Gmail，不是 GitHub noreply | 全部 278 個 commit 的 metadata | 接受；或公開時改用新 repo（squash 匯出）。之後的 commit 要不要改用 noreply email，由使用者決定（git config） |
| 2 | 兩台機器的使用者名稱出現在絕對路徑：`/Users/liujingwen/...`、`/home/wenn0113/...` | 約 2,000 個檔（幾乎都在 `results/` 的 env、rows、session、log）；`/home/wenn0113` 另見 MACHINE_POLICY、STORAGE_PLAN、`service/deploy/rtx.toml`、`scripts/wsl_*.sh`、部分 tests | 接受；或改用新 repo。已凍結的 result 不改。HEAD 的程式、設定與文件可在結案時改用 `~` / `$HOME` |
| 3 | 同機器上其他 private 專案的名稱（Incident Investigator、Rust HNSW、Text-to-SQL、agent-gym） | 9 個檔 13 處，加上 NEXT 歷史版本與 commit message | 接受，或在結案時從 HEAD 拿掉（history 仍在） |
| 4 | 私人基礎設施與桌面軟體的描述：Tailscale + SSH、免密碼的 alias、WSL distro、lock 路徑；Windows 端開著的程式與 VRAM（含遠端桌面軟體）、Mac 前景程式 | MACHINE_POLICY、`tools/rtx/*.sh`、031、012、NEXT 等 | 沒有可定位的位址。可接受，或結案時把 HEAD 的描述改得籠統一些 |
| 5 | 沒有 LICENSE；pyproject 也沒有宣告 | 全部 history | 公開前由使用者選授權（沒有授權時，外人在法律上不能重用程式碼） |
| 6 | 第三方授權只寫在 005（中文）：WikiText CC BY-SA 3.0 + GFDL、ARC CC BY-SA 4.0、MMLU MIT、WinoGrande CC BY、HellaSwag 經 ikawrakow 鏡像（005 記錄官方 repo 自 2026-09-14 起被 DMCA 封鎖）；Qwen3.5 在 pins 記為 apache-2.0，RedHatAI w4a16 沒有記錄授權 | README、SPEC、PREREGISTRATION 都沒有提到 | 結案時在 README 加入資料集與模型的來源與授權說明；HellaSwag 鏡像來源的說明由使用者決定怎麼寫 |
| 7 | 內部工作筆記：CLAUDE.md、NEXT.md、MACHINE_POLICY.md、docs/STORAGE_PLAN.md、docs/EXPERIMENT_RULES.md；commit message 中有額度暫停、使用者決定、無效嘗試等紀錄 | 從第一個 commit 起都在 | 接受（如實呈現 AI 協作的過程），或結案時從 HEAD 移除或改寫。commit message 無法改 |
| 8 | README 過時（狀態停在 2026-10-23、decision 只列到 041）；所有文件都是繁體中文，沒有英文摘要 | README.md | 結案時更新；要不要加英文摘要由使用者決定 |
| 9 | 只在本機的 ref（不在 origin） | 本機 `.git` | 公開時只 push main，不用 `--all` / `--mirror` |

## 結案時的待辦（照使用者的決定執行）

1. 使用者決定：直接公開這個 repo（接受第 1–4、7 項留在 history），還是改用新 repo（squash 匯出，沒有舊 history）。
2. 使用者用 secret scanner 掃完整 history。
3. 使用者選 LICENSE。
4. 我更新 README（結果、資料集與模型的來源與授權、重現步驟），並依使用者的決定整理第 2–4、7 項在 HEAD 的部分。
5. 公開那一步由使用者同意後才執行。

## 結案時的處理（2026-11-01，decision 052）

- 使用者的決定：直接公開這個 repo、保留完整 history（接受第 1–4、7 項留在 history，HEAD 也不另外改寫）；授權為 MIT（`LICENSE`、`pyproject.toml`）；
  README 首頁加中英對照的摘要，並寫入模型與資料集的來源與授權（第 6 項；HellaSwag 照 005 的原文寫）；第 8 項因此處理完。
- **重查 `2c5fddf` 之後新增的內容**（35 個 commit 加上結案的未 commit 變更，共 688 個檔；同樣的方向，依 MACHINE_POLICY 不搜尋 credential）：沒有會擋公開的項目。
  - 沒有題目原文（新增的 results/ 檔中含 5 個以上英文單字的字串只有 21 種，都是 harness 的說明文字）、模型檔或 tokenizer、`.gitignore` 類別的檔、> 1 MB 的 blob（最大 352,104 B）、刪除或改名。
  - 沒有新的 IP（只有 127.0.0.1）、MAC、序號、UUID（唯一的 UUID 是暫存目錄路徑的一段，屬於已接受的路徑類別）、主機名稱、他人姓名或 email，也沒有其他 private 專案的名稱。
  - 已接受的類別只增加數量：`/Users/liujingwen` 出現在 787 個檔、`/home/wenn0113` 在 1,749 個檔，新增的都在 `results/`。
  - origin 只有 `main`；本機另有兩個已併入 main 的 feat 分支，公開時不 push（第 9 項）。
- **使用者的 secret scan**（2026-11-01，`gitleaks git -v`，312 個 commit，也就是全部 315 個扣掉 3 個 merge）：5 筆 `generic-api-key`，都是誤報。
  被標記的值是 Qwen3.5-4B / 9B 公開 tokenizer 檔（`tokenizer.json`、`tokenizer_config.json`）的 SHA-256，記錄在模型來源的 checksum 清單中（只有 2 個不同的值）；
  規則是因為欄位名稱含「token」而觸發。5 個 fingerprint 寫進 `.gitleaksignore` 並註明原因。
- **公開**（2026-11-01，使用者同意後）：`jingwenliu0113-afk/llm-serving-lab` 改為 public；公開前確認工作樹乾淨，origin 只有 `main`（`57e0434`），沒有 tag。GitHub 辨識授權為 MIT。
- **公開後與 inspection-robot 專案對齊**（使用者 2026-11-01 決定）：
  - 之後的 commit 改用 GitHub noreply email（只改本 repo 的 `git config user.email`；舊 commit 不變，第 1 項）；
  - GitHub 的 repo 說明填上一句英文描述；
  - 加上 GitHub Actions CI（`.github/workflows/ci.yml`：ubuntu-24.04、uv、Python 3.12，抓完整 history，每次 push 跑全套測試；沒有 GPU、模型與資料）。
  - 沒有改：授權維持 MIT（robot 為 BSD-3-Clause）；README 維持一份（不拆成英文為主加 `README.zh-TW.md`）。

