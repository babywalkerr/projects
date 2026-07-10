import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = process.cwd();
const outputDir = path.join(root, "outputs", "error_analysis");
const qaDir = path.join(outputDir, "qa");
const summaryPath = path.join(outputDir, "metric_discrepancy_summary.json");

const summary = JSON.parse(await fs.readFile(summaryPath, "utf8"));

const workbook = Workbook.create();

function styleTitle(range) {
  range.format = {
    fill: "#111827",
    font: { bold: true, color: "#FFFFFF", size: 14 },
  };
}

function styleHeader(range) {
  range.format = {
    fill: "#1F2937",
    font: { bold: true, color: "#FFFFFF" },
    borders: { preset: "bottom", style: "thin", color: "#9CA3AF" },
  };
}

function styleBlock(range) {
  range.format = {
    fill: "#F9FAFB",
    borders: { preset: "outside", style: "thin", color: "#D1D5DB" },
  };
}

function pct(value) {
  return Math.round(value * 10000) / 100;
}

function countCsvRows(csvText) {
  let count = 0;
  let inQuotes = false;
  for (let i = 0; i < csvText.length; i += 1) {
    const ch = csvText[i];
    if (ch === '"') {
      if (csvText[i + 1] === '"') {
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
    } else if ((ch === "\n" || ch === "\r") && !inQuotes) {
      if (ch === "\r" && csvText[i + 1] === "\n") {
        i += 1;
      }
      count += 1;
    }
  }
  if (csvText.trim().length > 0 && !csvText.endsWith("\n") && !csvText.endsWith("\r")) {
    count += 1;
  }
  return Math.max(count, 1);
}

function parseCsv(csvText) {
  const rows = [];
  let row = [];
  let cell = "";
  let inQuotes = false;

  for (let i = 0; i < csvText.length; i += 1) {
    const ch = csvText[i];

    if (ch === '"') {
      if (inQuotes && csvText[i + 1] === '"') {
        cell += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }

    if (ch === "," && !inQuotes) {
      row.push(cell);
      cell = "";
      continue;
    }

    if ((ch === "\n" || ch === "\r") && !inQuotes) {
      if (ch === "\r" && csvText[i + 1] === "\n") {
        i += 1;
      }
      row.push(cell);
      if (row.some((value) => value.length > 0)) {
        rows.push(row);
      }
      row = [];
      cell = "";
      continue;
    }

    cell += ch;
  }

  row.push(cell);
  if (row.some((value) => value.length > 0)) {
    rows.push(row);
  }

  return rows;
}

function normalizeCsvRows(rows) {
  if (rows.length === 0) {
    return rows;
  }

  const numericColumns = new Set(["dataset_index", "label", "prediction", "score", "threshold"]);
  const headers = rows[0];

  return rows.map((row, rowIndex) => {
    if (rowIndex === 0) {
      return row;
    }

    return row.map((value, colIndex) => {
      if (!numericColumns.has(headers[colIndex])) {
        return value;
      }

      const numericValue = Number(value);
      return Number.isFinite(numericValue) ? numericValue : value;
    });
  });
}

const summarySheet = workbook.worksheets.add("Summary");
summarySheet.showGridLines = false;

summarySheet.getRange("A1:H1").merge();
summarySheet.getRange("A1:H1").values = [["Анализ расхождения метрик и ошибок модели"]];
styleTitle(summarySheet.getRange("A1:H1"));

summarySheet.getRange("A3:D3").values = [["Что сравнивали", "F1 toxic", "ROC-AUC", "Комментарий"]];
styleHeader(summarySheet.getRange("A3:D3"));

summarySheet.getRange("A4:D7").values = [
  [
    "18 экспериментов",
    summary.best_18_experiment.f1_toxic,
    summary.best_18_experiment.roc_auc,
    "Сэмпл 12 000 строк, порог 0.5, не финальная production-модель",
  ],
  [
    "Production layer 2, held-out",
    summary.production_layer2_heldout_recomputed.f1_toxic,
    summary.production_layer2_heldout_recomputed.roc_auc,
    "Честный отложенный тест 10% полного датасета",
  ],
  [
    "Цепочка без LLM, held-out",
    summary.chain_no_llm_heldout_recomputed.f1_toxic,
    summary.chain_no_llm_heldout_recomputed.roc_auc,
    "Словарь + layer 2 на том же отложенном тесте",
  ],
  [
    "Цепочка без LLM, full dataset",
    summary.chain_no_llm_full_dataset_previous.f1_toxic,
    summary.chain_no_llm_full_dataset_previous.roc_auc,
    "Нестрогая проверка: включает train-строки",
  ],
];
styleBlock(summarySheet.getRange("A4:D7"));
summarySheet.getRange("B4:C7").setNumberFormat("0.0000");
summarySheet.getRange("D4:D7").format.wrapText = true;
summarySheet.getRange("A4:H7").format.rowHeight = 38;

summarySheet.getRange("A10:F10").values = [["Модель", "TP", "TN", "FP", "FN", "Ошибок"]];
styleHeader(summarySheet.getRange("A10:F10"));
summarySheet.getRange("A11:F12").values = [
  [
    "Production layer 2 held-out",
    summary.production_layer2_heldout_recomputed.tp,
    summary.production_layer2_heldout_recomputed.tn,
    summary.production_layer2_heldout_recomputed.fp,
    summary.production_layer2_heldout_recomputed.fn,
    summary.production_layer2_heldout_recomputed.fp + summary.production_layer2_heldout_recomputed.fn,
  ],
  [
    "Цепочка без LLM held-out",
    summary.chain_no_llm_heldout_recomputed.tp,
    summary.chain_no_llm_heldout_recomputed.tn,
    summary.chain_no_llm_heldout_recomputed.fp,
    summary.chain_no_llm_heldout_recomputed.fn,
    summary.chain_no_llm_heldout_recomputed.fp + summary.chain_no_llm_heldout_recomputed.fn,
  ],
];
styleBlock(summarySheet.getRange("A11:F12"));
summarySheet.getRange("B11:F12").setNumberFormat("#,##0");

summarySheet.getRange("A15:B15").values = [["Стадия цепочки", "Количество комментариев"]];
styleHeader(summarySheet.getRange("A15:B15"));
const stageRows = Object.entries(summary.chain_no_llm_heldout_recomputed.stage_counts).map(([stage, count]) => [
  stage,
  count,
]);
summarySheet.getRangeByIndexes(15, 0, stageRows.length, 2).values = stageRows;
styleBlock(summarySheet.getRangeByIndexes(15, 0, stageRows.length, 2));
summarySheet.getRangeByIndexes(15, 1, stageRows.length, 1).setNumberFormat("#,##0");

summarySheet.getRange("F3:H7").values = [
  ["Короткий вывод", "", ""],
  ["0.9676 нельзя ставить как главную test-метрику: она посчитана по всему датасету.", "", ""],
  ["Основная честная метрика layer 2: F1 = 0.9365.", "", ""],
  ["Основная честная метрика цепочки без LLM: F1 = 0.9392.", "", ""],
  ["CSV-листы справа содержат реальные комментарии, где модель ошиблась.", "", ""],
];
summarySheet.getRange("F3:H3").merge();
summarySheet.getRange("F4:H7").merge(true);
styleHeader(summarySheet.getRange("F3:H3"));
styleBlock(summarySheet.getRange("F4:H7"));
summarySheet.getRange("F4:H7").format.wrapText = true;

summarySheet.getRange("A1:H20").format.font = { name: "Aptos" };
summarySheet.getRange("A1:A20").format.columnWidth = 30;
summarySheet.getRange("B1:C20").format.columnWidth = 16;
summarySheet.getRange("D1:D20").format.columnWidth = 54;
summarySheet.getRange("E1:E20").format.columnWidth = 16;
summarySheet.getRange("F1:H20").format.columnWidth = 32;
summarySheet.getRange("A10:F12").format.rowHeight = 26;
summarySheet.getRange("A15:B19").format.rowHeight = 24;
summarySheet.freezePanes.freezeRows(3);

async function importCsvSheet(sheetName, fileName, tableName, textColumnIndex) {
  const csvPath = path.join(outputDir, fileName);
  const csvText = await fs.readFile(csvPath, "utf8");
  const rows = normalizeCsvRows(parseCsv(csvText));

  const sheet = workbook.worksheets.add(sheetName);
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(1);

  const headers = rows[0];
  const rowCount = rows.length;
  const colCount = headers.length;
  const used = sheet.getRangeByIndexes(0, 0, rowCount, colCount);
  const header = sheet.getRangeByIndexes(0, 0, 1, colCount);
  used.values = rows;

  styleHeader(header);
  used.format.font = { name: "Aptos", size: 10 };
  used.format.borders = {
    insideHorizontal: { style: "thin", color: "#E5E7EB" },
    bottom: { style: "thin", color: "#D1D5DB" },
  };

  if (rowCount > 1) {
    const rangeAddress = `A1:${String.fromCharCode(64 + colCount)}${rowCount}`;
    const table = sheet.tables.add(rangeAddress, true, tableName);
    table.showFilterButton = true;
    table.style = "TableStyleMedium2";
  }

  sheet.getRangeByIndexes(0, 0, rowCount, 1).format.columnWidth = 14;
  sheet.getRangeByIndexes(0, 1, rowCount, Math.min(3, colCount - 1)).format.columnWidth = 13;
  sheet.getRangeByIndexes(0, 4, rowCount, Math.min(3, colCount - 4)).format.columnWidth = 16;
  sheet.getRangeByIndexes(0, textColumnIndex, rowCount, 1).format.columnWidth = 85;
  sheet.getRangeByIndexes(1, textColumnIndex, Math.max(rowCount - 1, 1), 1).format.wrapText = true;
  sheet.getRangeByIndexes(1, textColumnIndex, Math.min(Math.max(rowCount - 1, 1), 200), 1).format.rowHeight = 48;

  const scoreIndex = headers.indexOf("score");
  if (scoreIndex >= 0) {
    sheet.getRangeByIndexes(1, scoreIndex, Math.max(rowCount - 1, 1), 1).setNumberFormat("0.0000");
  }
}

await importCsvSheet("Prod all errors", "production_layer2_heldout_errors.csv", "ProdAllErrors", 6);
await importCsvSheet("Prod false positive", "production_layer2_false_positives.csv", "ProdFalsePositive", 6);
await importCsvSheet("Prod false negative", "production_layer2_false_negatives.csv", "ProdFalseNegative", 6);
await importCsvSheet("Chain all errors", "chain_no_llm_heldout_errors.csv", "ChainAllErrors", 7);
await importCsvSheet("Chain false positive", "chain_no_llm_false_positives.csv", "ChainFalsePositive", 7);
await importCsvSheet("Chain false negative", "chain_no_llm_false_negatives.csv", "ChainFalseNegative", 7);

await fs.mkdir(qaDir, { recursive: true });

const summaryPreview = await workbook.render({
  sheetName: "Summary",
  range: "A1:H20",
  scale: 1,
  format: "png",
});
await fs.writeFile(
  path.join(qaDir, "error_analysis_summary.png"),
  new Uint8Array(await summaryPreview.arrayBuffer()),
);

const errorsPreview = await workbook.render({
  sheetName: "Prod all errors",
  range: "A1:G20",
  scale: 1,
  format: "png",
});
await fs.writeFile(
  path.join(qaDir, "production_errors_preview.png"),
  new Uint8Array(await errorsPreview.arrayBuffer()),
);

const formulaErrors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 50 },
  summary: "formula error scan",
});
console.log(formulaErrors.ndjson);

const out = await SpreadsheetFile.exportXlsx(workbook);
const outputPath = path.join(outputDir, "error_analysis_report.xlsx");
await out.save(outputPath);
console.log(outputPath);
console.log(`Best 18 F1: ${pct(summary.best_18_experiment.f1_toxic)}%`);
console.log(`Production held-out F1: ${pct(summary.production_layer2_heldout_recomputed.f1_toxic)}%`);
console.log(`Chain held-out F1: ${pct(summary.chain_no_llm_heldout_recomputed.f1_toxic)}%`);
