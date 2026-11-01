# 052 — Phase 3 結案：正式設定不變、公開方式與授權

- 狀態：**Accepted**（使用者 2026-11-01 決定）
- 日期：2026-11-01
- 對應：037（Phase 3 的範圍與共通規則）、051（Stage V 的結果與「使用者決定」）、036（Phase 2 結案的寫法）、037 §2（改釘選的程序）、`docs/PUBLIC_RELEASE_AUDIT.md`

## 背景

- Phase 3 的範圍（037）都已完成，每一項都依各自凍結的規則判定：
  - Stage D（039 / 042）與它的產品化（043）；
  - Stage E：E1-D（045）、E1-P（046）、E3（047）、E2（050）；
  - Stage Q：Q1 / Q2（048）、Q3（049）；
  - 產品 v1（038）；
  - Stage V（044 / 051）：產品化的驗證研究。
- 研究的 GPU 工作全部結束。Stage V 的全期研究時數 7.44 h（上限 12 h）。
- 051 列出要使用者決定的事：是否改正式釘選、是否開啟 PACE-8K（等於改釘選）。
- 使用者 2026-10-30 說：專題完成後 repo 要在 GitHub 公開，作為作品集。公開前的唯讀預查見 `docs/PUBLIC_RELEASE_AUDIT.md`，其中「結案時的待辦」要使用者決定公開方式與授權。

## 決策（使用者 2026-11-01）

1. **正式設定維持現狀**：`service/deploy/rtx-verified.toml` 仍釘住 `verified-formal-20261017-c1b.json`，active class 為 uniform-2k（32）。
   - 不改釘選、不開啟 PACE-8K、不常駐。
   - Stage D 的三份與 Stage V 的兩份 verified profile 都只 commit、不釘選。
2. **Phase 3 結案**：V3（經過 gateway 的開放迴路）、T 的部署與 V5、常駐，留在 README 的 Future Work，本專題不執行。
3. **公開方式**：直接公開這個 repo，保留完整 history。
   - 接受 `PUBLIC_RELEASE_AUDIT.md` 第 1–4、7 項留在 history：commit metadata 的個人 email、兩台機器的使用者名稱（絕對路徑）、同機器上其他 private 專案的名稱、私人基礎設施的描述、內部工作筆記。HEAD 也不另外改寫這些項目。
   - 公開時只有 main 在 origin（第 9 項）。
4. **授權**：MIT（`LICENSE`，`pyproject.toml` 的 `license = "MIT"`）。
5. **README**：首頁加中英對照的摘要（做了什麼、重要發現、方法、技術、怎麼看這個 repo），以及模型與資料集的來源與授權（第 6 項）；原本的詳細內容保留在下方。`service/README.md` 加上 Stage V 的 profile 一節。

## 為什麼

- 公開的目的是作品集。改釘選或開啟 PACE-8K 不會增加任何證據，卻要再跑一次 RTX 產品測試（IT-0–6）；PACE-8K 也只證明 pacing 開啟時穩態仍符合 SLO，沒有改善（051）。
- 保留 history：本專題的核心是「事先登記、先凍結再看數據」，它的證據就是 commit 的時間順序（凍結 commit 早於正式 session 的 commit）。squash 成新 repo 會失去這項證據與開發的軌跡。
- MIT 是最常見、最簡單的寬鬆授權。

## 放棄的替代方案

- 改正式釘選（uniform-32k、mixed-8k-t、`uniform-2k-o1-512`、`uniform-8k-o1-512`）或開啟 PACE-8K：使用者沒有選；兩者都要改設定並重跑 IT-0–6，mixed-8k-t 另需 T 的部署與 V5。
- 另開新 repo（squash 匯出，沒有舊 history）：可以拿掉 history 中的個人資訊，但失去凍結先於數據的時間證據。
- Apache-2.0（多了專利條款）、不放 LICENSE（別人在法律上不能重用程式碼）。
- 只用中文的摘要、或不改 README。

## 對實驗的影響

- 沒有。不改任何結果、判定、規則、凍結或正式設定；只加 LICENSE、授權宣告與文件。
- 公開前剩下的步驟：
  1. 重查 `PUBLIC_RELEASE_AUDIT.md` 之後新增的 commit（同樣的方向；依 MACHINE_POLICY 不搜尋 credential）；
  2. 使用者自己用 secret scanner（例如 gitleaks）掃完整 history；
  3. 使用者同意後，才把 repo 改為 public。
