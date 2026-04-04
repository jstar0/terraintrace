import type { EChartsCoreOption as EChartsOption } from "echarts/core";
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import type {
  ChangeEvent,
  KeyboardEvent as ReactKeyboardEvent,
  MouseEvent as ReactMouseEvent,
  PointerEvent as ReactPointerEvent,
  RefObject,
  WheelEvent as ReactWheelEvent,
} from "react";
import { createRoot } from "react-dom/client";
import "./style.css";

type Mode = "benchmark" | "inference";
type Split = "train" | "validation" | "test" | "inference";
type DatasetCase = {
  id: string;
  group?: string;
  split: Split;
  scale: number;
  pixel_size: number;
  elevation_unit: string;
  shape: number[];
  display_name?: string;
};
type DatasetRecord = {
  dataset_id: string;
  revision: number;
  mode: Mode;
  manifest: { mode: Mode; cases: DatasetCase[] };
};
type Metric = {
  rmse: number;
  mae: number;
  slopeMae: number;
  gradientCorrelation: number | null;
  gradientCorrelationReason: string | null;
  edgeRmse: number;
  lowResolutionRmse: number;
  areaConsistencyMax: number;
};
type ArrayMeta = {
  method: string;
  shape: number[];
  dtype: string;
  sha256: string;
  value_range: number[];
  low_resolution_rmse: number;
  path: string;
};
type RunCase = {
  case_id: string;
  group: string | null;
  split: Split;
  status: string;
  input_shape: number[];
  arrays: Record<string, ArrayMeta>;
  metrics: Record<string, Metric | null>;
};
type RunPayload = {
  status: string;
  mode: Mode;
  job_id: string;
  metrics_protocol: string;
  config: { methods: string[] };
  input_sha256?: string;
  config_sha256?: string;
  engine_version?: string;
  prior?: PriorMeta | null;
  cases: RunCase[];
};
type RunRecord = {
  run_id: string;
  dataset_id: string;
  dataset_revision: number;
  payload: RunPayload;
};
type Job = {
  job_id: string;
  run_id: string;
  status: string;
  progress: { done: number; total: number };
  error?: string | null;
};
type Grid = { rows: number; cols: number; values: Float64Array };
type ViewerState = { caseId: string; grids: Record<string, Grid>; loading: boolean; error?: string };
type Point = { row: number; col: number };
type Viewport = { zoom: number; panRows: number; panCols: number };
type ProfileData = { points: number[][]; samples: number; distance: number[]; series: Record<string, number[]> };
type PriorMeta = {
  bins: number;
  radius_centers: number[];
  gains: number[];
  clip: number[];
  train_case_hashes: string[];
  scale: number;
  compatible_input: { shape: number[]; pixel_size: number; elevation_unit: string };
};
type ChartInstance = { setOption: (option: EChartsOption) => void; resize: () => void; dispose: () => void };
type RoiData = {
  exploratory: true;
  bounds: number[];
  methods: Record<string, { min: number; max: number; mean: number }>;
};
type Session = { datasetId: string; revision: number; runId?: string };
type OverviewMetric = "rmse" | "mae" | "slopeMae" | "gradientCorrelation" | "edgeRmse" | "lr";

const API = "/api/v1";
const STORAGE_KEY = "terraintrace.session.v1";
const TERMINAL = new Set(["succeeded", "failed", "cancelled", "interrupted"]);
const ELEVATION_STOPS = ["#244c58", "#287e79", "#73aa8c", "#d6cf85", "#d88957"];
const ERROR_STOPS = ["#316987", "#a9cfca", "#f2f1e9", "#df9a8b", "#a94743"];
const SPLITS: { id: Split; label: string }[] = [
  { id: "train", label: "训练" },
  { id: "validation", label: "验证" },
  { id: "test", label: "测试" },
  { id: "inference", label: "仅推断" },
];

class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

async function apiJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, init);
  const text = await response.text();
  let body: unknown = {};
  try {
    body = text ? JSON.parse(text) : {};
  } catch {
    body = { message: text || response.statusText };
  }
  if (!response.ok) {
    const value = body as { error?: { message?: string }; detail?: { message?: string } | string; message?: string };
    const message = typeof value.detail === "object" ? value.detail.message : value.detail;
    throw new ApiError(response.status, value.error?.message ?? message ?? value.message ?? response.statusText);
  }
  return body as T;
}

async function fetchGrid(path: string): Promise<Grid> {
  const response = await fetch(`${API}${path}`);
  if (!response.ok) {
    const body = (await response.json().catch(() => ({}))) as { error?: { message?: string }; detail?: { message?: string } | string };
    const message = typeof body.detail === "object" ? body.detail?.message : body.detail;
    throw new ApiError(response.status, body.error?.message ?? message ?? "数组读取失败");
  }
  return parseNpy(await response.arrayBuffer());
}

function parseNpy(buffer: ArrayBuffer): Grid {
  const bytes = new Uint8Array(buffer);
  const magic = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59];
  if (!magic.every((byte, index) => bytes[index] === byte)) throw new Error("服务返回的文件不是 NPY 数组");
  const view = new DataView(buffer);
  const major = bytes[6];
  const headerLength = major === 1 ? view.getUint16(8, true) : view.getUint32(8, true);
  const headerStart = major === 1 ? 10 : 12;
  const header = new TextDecoder().decode(bytes.subarray(headerStart, headerStart + headerLength));
  const dtype = header.match(/['"]descr['"]\s*:\s*['"]([<>=|]?\w\d+)['"]/)?.[1];
  const shapeText = header.match(/['"]shape['"]\s*:\s*\(([^)]*)\)/)?.[1];
  const shape = shapeText?.split(",").map((value) => Number(value.trim())).filter(Number.isFinite) ?? [];
  if (!dtype || !["<f8", "=f8", "|f8"].includes(dtype) || shape.length !== 2) {
    throw new Error(`暂不支持此数组格式（${dtype ?? "dtype 未知"}）`);
  }
  const rows = shape[0];
  const cols = shape[1];
  const dataStart = headerStart + headerLength;
  if (rows < 1 || cols < 1 || dataStart + rows * cols * 8 > buffer.byteLength) {
    throw new Error("NPY 数组头或数据长度不匹配");
  }
  const values = new Float64Array(rows * cols);
  const littleEndian = !dtype.startsWith(">") && !(dtype.startsWith("=") && /big/i.test(navigator.userAgent));
  for (let index = 0; index < values.length; index += 1) {
    values[index] = view.getFloat64(dataStart + index * 8, littleEndian);
  }
  return { rows, cols, values };
}

function safeId(value: string): string {
  return encodeURIComponent(value);
}

function methodPath(runId: string, caseId: string, method: string, kind = "reconstruction"): string {
  const params = new URLSearchParams({ method, kind });
  return `/runs/${safeId(runId)}/cases/${safeId(caseId)}/arrays?${params.toString()}`;
}

function requestJson(body: unknown): RequestInit {
  return { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
}

let echartsRuntime: Promise<typeof import("echarts/core")> | null = null;

function loadEchartsCore(): Promise<typeof import("echarts/core")> {
  if (!echartsRuntime) {
    echartsRuntime = Promise.all([
      import("echarts/core"),
      import("echarts/charts"),
      import("echarts/components"),
      import("echarts/renderers"),
    ]).then(([core, charts, components, renderers]) => {
      core.use([charts.LineChart, components.GridComponent, components.LegendComponent, components.TooltipComponent, renderers.CanvasRenderer]);
      return core;
    });
  }
  return echartsRuntime;
}

async function createChart(element: HTMLDivElement, options: EChartsOption): Promise<ChartInstance> {
  const core = await loadEchartsCore();
  const chart = core.init(element);
  chart.setOption(options);
  return chart;
}

function expandLowResolution(low: Grid, scale: number, rows: number, cols: number): Grid {
  if (low.rows * scale !== rows || low.cols * scale !== cols) {
    throw new Error("低分辨率网格与重建尺寸不匹配，无法放入相同坐标系");
  }
  const values = new Float64Array(rows * cols);
  for (let row = 0; row < rows; row += 1) {
    const sourceRow = Math.floor(row / scale);
    for (let col = 0; col < cols; col += 1) {
      values[row * cols + col] = low.values[sourceRow * low.cols + Math.floor(col / scale)];
    }
  }
  return { rows, cols, values };
}

function gridRange(grids: Grid[]): [number, number] {
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  for (const grid of grids) {
    for (const value of grid.values) {
      if (value < min) min = value;
      if (value > max) max = value;
    }
  }
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [0, 1];
  return min === max ? [min - 0.5, max + 0.5] : [min, max];
}

function formatNumber(value: number | null | undefined, digits = 4): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return value.toFixed(digits);
}

function App() {
  const [dataset, setDataset] = useState<DatasetRecord | null>(null);
  const [run, setRun] = useState<RunRecord | null>(null);
  const [runCases, setRunCases] = useState<RunCase[]>([]);
  const [job, setJob] = useState<Job | null>(null);
  const [selectedSplit, setSelectedSplit] = useState<Split>("test");
  const [selectedCaseId, setSelectedCaseId] = useState<string | null>(null);
  const [selectedMethods, setSelectedMethods] = useState<string[]>([]);
  const [tab, setTab] = useState<"case" | "overview">("case");
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);
  const [showImport, setShowImport] = useState(false);
  const [importMode, setImportMode] = useState<Mode>("benchmark");
  const [files, setFiles] = useState<File[]>([]);
  const [metadataText, setMetadataText] = useState("");
  const [viewer, setViewer] = useState<ViewerState | null>(null);
  const [panels, setPanels] = useState<string[]>(["high_res", "method:bilinear", "method:spectral_prior"]);
  const [mobilePanel, setMobilePanel] = useState(0);
  const [viewport, setViewport] = useState<Viewport>({ zoom: 1, panRows: 0, panCols: 0 });
  const [cursor, setCursor] = useState<Point | null>(null);
  const [profilePoints, setProfilePoints] = useState<{ A: Point | null; B: Point | null }>({ A: null, B: null });
  const [profileFocus, setProfileFocus] = useState<"A" | "B">("A");
  const [placingPoint, setPlacingPoint] = useState<"A" | "B" | null>(null);
  const [profileData, setProfileData] = useState<ProfileData | null>(null);
  const [roiSelection, setRoiSelection] = useState<{ start: Point; end: Point } | null>(null);
  const [roiData, setRoiData] = useState<RoiData | null>(null);
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number; point: Point } | null>(null);
  const [exportArtifactId, setExportArtifactId] = useState<string | null>(null);
  const [overviewMetric, setOverviewMetric] = useState<OverviewMetric>("rmse");
  const [overviewSortMethod, setOverviewSortMethod] = useState("bilinear");
  const [sortDescending, setSortDescending] = useState(false);
  const [caseFilter, setCaseFilter] = useState("");
  const [booting, setBooting] = useState(true);
  const fileInput = useRef<HTMLInputElement>(null);
  const chartRef = useRef<HTMLDivElement>(null);
  const eventSource = useRef<EventSource | null>(null);

  const activeCase = runCases.find((row) => row.case_id === selectedCaseId) ?? null;
  const splitCases = useMemo(
    () => runCases.filter((row) => row.split === selectedSplit),
    [runCases, selectedSplit],
  );
  const filteredCases = useMemo(
    () => splitCases.filter((row) => `${row.case_id} ${row.group ?? ""}`.toLowerCase().includes(caseFilter.toLowerCase())),
    [splitCases, caseFilter],
  );
  const caseMeta = dataset?.manifest.cases.find((row) => row.id === selectedCaseId) ?? null;
  const methodNames = activeCase ? Object.keys(activeCase.arrays) : [];
  const caseMetrics = activeCase?.metrics ?? {};
  const activeGrid = panels[mobilePanel] ? viewer?.grids[panels[mobilePanel]] : undefined;

  const loadRun = useCallback(async (runId: string) => {
    const [runRecord, casesRecord] = await Promise.all([
      apiJson<RunRecord>(`/runs/${safeId(runId)}`),
      apiJson<{ status: string; cases: RunCase[] }>(`/runs/${safeId(runId)}/cases`),
    ]);
    setRun(runRecord);
    setRunCases(casesRecord.cases);
    setOverviewMetric(runRecord.payload.mode === "inference" ? "lr" : "rmse");
    setOverviewSortMethod(runRecord.payload.config.methods.includes("bilinear") ? "bilinear" : runRecord.payload.config.methods[0]);
    const first = casesRecord.cases.find((row) => row.split === "test") ?? casesRecord.cases[0];
    if (first) {
      setSelectedSplit(first.split);
      setSelectedCaseId(first.case_id);
    }
    setJob((current) => current?.job_id === runRecord.payload.job_id ? current : null);
    return runRecord;
  }, []);

  const restoreSession = useCallback(async (session: Session) => {
    const record = await apiJson<DatasetRecord>(`/terrain-datasets/${safeId(session.datasetId)}/${session.revision}`);
    setDataset(record);
    setSelectedSplit(record.mode === "benchmark" ? "test" : "inference");
    setSelectedMethods(record.mode === "benchmark"
      ? ["nearest", "bilinear", ...(record.manifest.cases.some((row) => row.split === "train") ? ["spectral_prior"] : [])]
      : ["nearest", "bilinear"]);
    if (session.runId) {
      const restored = await loadRun(session.runId);
      setOverviewMetric(restored.payload.mode === "inference" ? "lr" : "rmse");
      setJob(await apiJson<Job>(`/jobs/${safeId(restored.payload.job_id)}`));
    }
  }, [loadRun]);

  useEffect(() => {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (raw) {
      try {
        void restoreSession(JSON.parse(raw) as Session).catch(() => {
          window.localStorage.removeItem(STORAGE_KEY);
        }).finally(() => setBooting(false));
        return;
      } catch {
        window.localStorage.removeItem(STORAGE_KEY);
      }
    }
    setBooting(false);
  }, [restoreSession]);

  useEffect(() => {
    if (!activeCase || !run || !selectedCaseId || !dataset) {
      setViewer(null);
      return;
    }
    let current = true;
    const currentCaseId = selectedCaseId;
    setViewer({ caseId: currentCaseId, grids: {}, loading: true });
    setProfileData(null);
    setRoiData(null);
    setProfilePoints({ A: null, B: null });
    setViewport({ zoom: 1, panRows: 0, panCols: 0 });
    setCursor(null);
    const read = async () => {
      const prefix = `/runs/${safeId(run.run_id)}/cases/${safeId(currentCaseId)}/arrays`;
      const grids: Record<string, Grid> = {};
      grids.low_res = await fetchGrid(`${prefix}?kind=low_res`);
      if (run.payload.mode === "benchmark") grids.high_res = await fetchGrid(`${prefix}?kind=high_res`);
      const methods = Object.keys(activeCase.arrays);
      const baseRows = await Promise.all(methods.map(async (method) => [
        `method:${method}`,
        await fetchGrid(methodPath(run.run_id, currentCaseId, method)),
      ] as const));
      for (const [key, grid] of baseRows) grids[key] = grid;
      if (run.payload.mode === "benchmark") {
        const errorRows = await Promise.all(methods.flatMap((method) => [
          fetchGrid(methodPath(run.run_id, currentCaseId, method, "difference")).then((grid) => [`difference:${method}`, grid] as const),
          fetchGrid(methodPath(run.run_id, currentCaseId, method, "absolute_error")).then((grid) => [`absolute_error:${method}`, grid] as const),
        ]));
        for (const [key, grid] of errorRows) grids[key] = grid;
      }
      const expected = run.payload.mode === "benchmark"
        ? ["high_res", "method:bilinear", `method:${methods.includes("spectral_prior") ? "spectral_prior" : methods[0]}`]
        : ["low_res", "method:bilinear", "method:nearest"];
      if (current) {
        setViewer({ caseId: currentCaseId, grids, loading: false });
        setPanels(expected.map((key) => grids[key] ? key : methods.length ? `method:${methods[0]}` : "low_res"));
        setMobilePanel(0);
        const baseGrid = grids.high_res ?? grids["method:bilinear"] ?? grids.low_res;
        setProfilePoints({ A: { row: 0, col: 0 }, B: { row: baseGrid.rows - 1, col: baseGrid.cols - 1 } });
      }
    };
    void read().catch((error: unknown) => {
      if (current) setViewer({ caseId: currentCaseId, grids: {}, loading: false, error: error instanceof Error ? error.message : "读取案例数据失败" });
    });
    return () => { current = false; };
  }, [activeCase, dataset, run, selectedCaseId]);

  useEffect(() => {
    const source = eventSource.current;
    source?.close();
    eventSource.current = null;
    if (!job || TERMINAL.has(job.status)) return;
    const stream = new EventSource(`${API}/jobs/${safeId(job.job_id)}/events?after=0`);
    eventSource.current = stream;
    let finished = false;
    const refresh = async (status?: string, progress?: Job["progress"]) => {
      if (status && progress) setJob((old) => old ? { ...old, status, progress } : old);
      try {
        const latest = await apiJson<Job>(`/jobs/${safeId(job.job_id)}`);
        setJob(latest);
        if (TERMINAL.has(latest.status) && !finished) {
          finished = true;
          stream.close();
          eventSource.current = null;
          await loadRun(job.run_id);
        }
      } catch (error) {
        setNotice(error instanceof Error ? `无法读取任务进度：${error.message}` : "无法读取任务进度");
      }
    };
    const onEvent = (event: Event) => {
      const message = event as MessageEvent<string>;
      const type = message.type;
      try {
        const payload = JSON.parse(message.data) as { done?: number; total?: number; status?: string };
        void refresh(type === "progress" ? payload.status : type, {
          done: payload.done ?? job.progress.done,
          total: payload.total ?? job.progress.total,
        });
      } catch {
        void refresh();
      }
    };
    for (const type of ["queued", "running", "progress", "cancelling", "cancelled", "succeeded", "failed", "interrupted"]) {
      stream.addEventListener(type, onEvent);
    }
    stream.onerror = () => { void refresh(); };
    const poll = window.setInterval(() => { void refresh(); }, 900);
    return () => {
      window.clearInterval(poll);
      stream.close();
      if (eventSource.current === stream) eventSource.current = null;
    };
  }, [job?.job_id, loadRun]);

  useEffect(() => {
    const element = chartRef.current;
    if (!element || !profileData) return;
    let chart: ChartInstance | null = null;
    let cancelled = false;
    const options: EChartsOption = {
      animation: false,
      grid: { top: 18, right: 20, bottom: 34, left: 52 },
      tooltip: { trigger: "axis", valueFormatter: (value: unknown) => `${Number(value).toFixed(3)} ${caseMeta?.elevation_unit ?? ""}` },
      legend: { top: 0, right: 8, textStyle: { color: "#536865", fontSize: 10 } },
      xAxis: { type: "value", name: "距离（声明像素间距单位）", nameLocation: "middle", nameGap: 26, axisLabel: { color: "#627774", fontSize: 10 }, axisLine: { lineStyle: { color: "#bcc9c6" } }, splitLine: { lineStyle: { color: "#e4eae8" } } },
      yAxis: { type: "value", name: caseMeta?.elevation_unit ?? "高程", axisLabel: { color: "#627774", fontSize: 10 }, axisLine: { lineStyle: { color: "#bcc9c6" } }, splitLine: { lineStyle: { color: "#e4eae8" } } },
      series: Object.entries(profileData.series).map(([name, values]) => ({
        name: name === "high_res" ? "真值" : name,
        type: "line",
        showSymbol: false,
        lineStyle: {
          width: name === "high_res" ? 2.5 : 1.8,
          color: name === "high_res" ? "#243b3a" : name === "bilinear" ? "#287e79" : name === "nearest" ? "#d88957" : "#477e9d",
        },
        data: profileData.distance.map((distance, pointIndex) => [distance, values[pointIndex]]),
      })),
    };
    const resize = () => chart?.resize();
    void createChart(element, options).then((created) => {
      if (cancelled) { created.dispose(); return; }
      chart = created;
      window.addEventListener("resize", resize);
    }).catch((error: unknown) => {
      if (!cancelled) setNotice(error instanceof Error ? `剖面图无法载入：${error.message}` : "剖面图无法载入");
    });
    return () => { cancelled = true; window.removeEventListener("resize", resize); chart?.dispose(); };
  }, [caseMeta, profileData]);

  useEffect(() => {
    if (!contextMenu) return;
    const close = (event: MouseEvent | KeyboardEvent) => {
      if (event instanceof KeyboardEvent && event.key !== "Escape") return;
      setContextMenu(null);
    };
    window.addEventListener("pointerdown", close);
    window.addEventListener("keydown", close);
    return () => {
      window.removeEventListener("pointerdown", close);
      window.removeEventListener("keydown", close);
    };
  }, [contextMenu]);

  const installDataset = async (id: string, revision: number) => {
    const next = await apiJson<DatasetRecord>(`/terrain-datasets/${safeId(id)}/${revision}`);
    setDataset(next);
    setRun(null);
    setRunCases([]);
    setJob(null);
    setSelectedCaseId(null);
    setSelectedSplit(next.mode === "benchmark" ? "test" : "inference");
    setExportArtifactId(null);
    setSelectedMethods(next.mode === "benchmark"
      ? ["nearest", "bilinear", ...(next.manifest.cases.some((row) => row.split === "train") ? ["spectral_prior"] : [])]
      : ["nearest", "bilinear"]);
    setViewer(null);
    setTab("case");
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify({ datasetId: id, revision } satisfies Session));
  };

  const openDemo = async () => {
    setBusy(true);
    setNotice("正在保存固定种子的合成示例…");
    try {
      const created = await apiJson<{ dataset_id: string; revision: number }>("/demo/datasets?seed=17", { method: "POST" });
      await installDataset(created.dataset_id, created.revision);
      setNotice("示例数据已准备好。组间隔离保持为 train / validation / test，可开始运行比较。");
    } catch (error) {
      setNotice(error instanceof Error ? `打开示例失败：${error.message}` : "打开示例失败");
    } finally {
      setBusy(false);
    }
  };

  const updateFiles = (event: ChangeEvent<HTMLInputElement>) => {
    const next = Array.from(event.target.files ?? []);
    setFiles(next);
    if (importMode === "benchmark") {
      setMetadataText(JSON.stringify(next.map((file, index) => ({
        id: file.name.replace(/\.npz$/i, "") || `case-${index + 1}`,
        group: `group-${index + 1}`,
        split: "test",
        scale: 4,
        pixel_size: 1,
        elevation_unit: "m",
      })), null, 2));
    } else {
      const first = next[0];
      setMetadataText(JSON.stringify({ id: first?.name.replace(/\.npz$/i, "") || "inference-case", scale: 4, pixel_size: 1, elevation_unit: "m" }, null, 2));
    }
  };

  const switchImportMode = (mode: Mode) => {
    setImportMode(mode);
    if (!files.length) return;
    setMetadataText(mode === "benchmark"
      ? JSON.stringify(files.map((file, index) => ({ id: file.name.replace(/\.npz$/i, "") || `case-${index + 1}`, group: `group-${index + 1}`, split: "test", scale: 4, pixel_size: 1, elevation_unit: "m" })), null, 2)
      : JSON.stringify({ id: files[0].name.replace(/\.npz$/i, "") || "inference-case", scale: 4, pixel_size: 1, elevation_unit: "m" }, null, 2));
  };

  const uploadDataset = async () => {
    if (!files.length) {
      setNotice("先选择一个或多个 NPZ 文件");
      return;
    }
    setBusy(true);
    setNotice("正在检查 NPZ、元数据与数据组隔离…");
    try {
      const form = new FormData();
      form.set("mode", importMode);
      form.set("metadata", metadataText);
      files.forEach((file) => form.append("files", file, file.name));
      const created = await apiJson<{ dataset_id: string; revision: number; cases: DatasetCase[] }>("/terrain-datasets", { method: "POST", body: form });
      await installDataset(created.dataset_id, created.revision);
      setShowImport(false);
      setFiles([]);
      setNotice(`导入完成：${created.cases.length} 个案例，版本 ${created.revision}。${importMode === "inference" ? "这是 LR-only 数据，没有 HR 真值。" : "数据仍按声明的 group/split 隔离。"}`);
    } catch (error) {
      setNotice(error instanceof Error ? `导入没有写入数据集：${error.message}` : "导入失败；请检查 NPZ 和元数据");
    } finally {
      setBusy(false);
    }
  };

  const toggleMethod = (method: string) => {
    setSelectedMethods((current) => current.includes(method) ? current.filter((item) => item !== method) : [...current, method]);
  };

  const startRun = async () => {
    if (!dataset) return;
    if (!selectedMethods.length) {
      setNotice("至少选择一种重建方法");
      return;
    }
    setBusy(true);
    setNotice("正在将运行配置固定到当前数据集版本…");
    setExportArtifactId(null);
    try {
      const created = await apiJson<{ run_id: string; job_id: string; status: string; dataset_revision: number }>("/reconstruction-runs", requestJson({
        dataset_id: dataset.dataset_id,
        dataset_revision: dataset.revision,
        mode: dataset.mode,
        methods: selectedMethods,
        metrics_protocol: "terrain_v2",
      }));
      const runRecord = await apiJson<RunRecord>(`/runs/${safeId(created.run_id)}`);
      setRun(runRecord);
      setRunCases([]);
      setSelectedCaseId(null);
      const initialJob = await apiJson<Job>(`/jobs/${safeId(created.job_id)}`);
      setJob(initialJob);
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify({ datasetId: dataset.dataset_id, revision: dataset.revision, runId: created.run_id } satisfies Session));
      setTab("case");
      if (TERMINAL.has(initialJob.status)) {
        await loadRun(created.run_id);
        setNotice(initialJob.status === "succeeded" ? "运行完成，已读取服务端数组和案例指标。" : `任务状态：${initialJob.status}`);
      } else {
        setNotice("任务已排队。进度来自实际完成的案例；可在案例边界取消。");
      }
    } catch (error) {
      setNotice(error instanceof Error ? `无法创建运行：${error.message}` : "无法创建运行");
    } finally {
      setBusy(false);
    }
  };

  const cancelRun = async () => {
    if (!job) return;
    try {
      const updated = await apiJson<Job>(`/jobs/${safeId(job.job_id)}/cancel`, { method: "POST" });
      setJob(updated);
      setNotice(updated.status === "cancelled" ? "排队任务已取消" : "正在结束当前案例；已经完成的结果会保留并标为未完成");
    } catch (error) {
      setNotice(error instanceof Error ? `取消失败：${error.message}` : "取消失败");
    }
  };

  const sampleProfile = async () => {
    if (!run || !activeCase || !profilePoints.A || !profilePoints.B) {
      setNotice("先在地图上设置剖面端点 A 和 B");
      return;
    }
    try {
      const result = await apiJson<ProfileData>(`/runs/${safeId(run.run_id)}/profiles`, requestJson({
        case_id: activeCase.case_id,
        points: [[profilePoints.A.row, profilePoints.A.col], [profilePoints.B.row, profilePoints.B.col]],
        methods: methodNames,
        samples: 128,
      }));
      setProfileData(result);
      setProfileFocus("A");
      setNotice("剖面由服务端按同一条路径采样；曲线和 CSV 使用相同 128 个点。");
    } catch (error) {
      setNotice(error instanceof Error ? `剖面生成失败：${error.message}` : "剖面生成失败");
    }
  };

  const downloadProfile = () => {
    if (!profileData) return;
    const methods = Object.keys(profileData.series);
    const rows = [
      ["sample", "distance", ...methods].join(","),
      ...profileData.distance.map((distance, index) => [index, distance, ...methods.map((method) => profileData.series[method][index])].join(",")),
    ];
    downloadBlob(new Blob([`${rows.join("\n")}\n`], { type: "text/csv;charset=utf-8" }), `terraintrace-${activeCase?.case_id ?? "profile"}-profile.csv`);
  };

  const computeRoi = async (bounds: number[]) => {
    if (!run || !activeCase) return;
    try {
      const result = await apiJson<RoiData>(`/runs/${safeId(run.run_id)}/roi`, requestJson({ case_id: activeCase.case_id, bounds, methods: methodNames }));
      setRoiData(result);
      setNotice("ROI 结果只作局部探索，保留在全案例指标之外。");
    } catch (error) {
      setNotice(error instanceof Error ? `ROI 无法计算：${error.message}` : "ROI 无法计算");
    }
  };

  const exportCase = async () => {
    if (!run || !activeCase) return;
    try {
      const result = await apiJson<{ artifact_id: string; bytes: number }>(`/runs/${safeId(run.run_id)}/exports?case_id=${safeId(activeCase.case_id)}`, { method: "POST" });
      setExportArtifactId(result.artifact_id);
      setNotice(`已生成并校验案例归档，${result.bytes.toLocaleString()} bytes。下载包包含数组、指标、剖面、说明和完整性清单。`);
    } catch (error) {
      setNotice(error instanceof Error ? `导出失败：${error.message}` : "导出失败");
    }
  };

  const selectProfilePoint = (which: "A" | "B", point: Point) => {
    setProfilePoints((current) => ({ ...current, [which]: point }));
    setProfileFocus(which);
    setPlacingPoint(null);
    setProfileData(null);
  };

  const nudgeProfile = (which: "A" | "B", rowDelta: number, colDelta: number) => {
    setProfilePoints((current) => {
      const point = current[which];
      if (!point || !activeGrid) return current;
      return {
        ...current,
        [which]: {
          row: Math.max(0, Math.min(activeGrid.rows - 1, point.row + rowDelta)),
          col: Math.max(0, Math.min(activeGrid.cols - 1, point.col + colDelta)),
        },
      };
    });
    setProfileData(null);
  };

  const benchmark = run?.payload.mode === "benchmark";
  const hasTraining = dataset?.manifest.cases.some((row) => row.split === "train") ?? false;
  const progress = job?.progress.total ? Math.min(100, job.progress.done / job.progress.total * 100) : 0;
  const mapGrids = useMemo(() => {
    const result = { ...(viewer?.grids ?? {}) };
    if (result.low_res && caseMeta && result.high_res) {
      result.low_res = expandLowResolution(result.low_res, caseMeta.scale, result.high_res.rows, result.high_res.cols);
    } else if (result.low_res && caseMeta) {
      const target = result["method:bilinear"] ?? result["method:nearest"];
      if (target) result.low_res = expandLowResolution(result.low_res, caseMeta.scale, target.rows, target.cols);
    }
    return result;
  }, [caseMeta, viewer]);
  const elevationRange = useMemo(() => gridRange(Object.entries(mapGrids).filter(([key]) => !key.includes("error") && !key.includes("difference")).map(([, grid]) => grid)), [mapGrids]);
  const errorMax = useMemo(() => {
    let max = 0;
    for (const [key, grid] of Object.entries(mapGrids)) {
      if (!key.startsWith("difference:") && !key.startsWith("absolute_error:")) continue;
      for (const value of grid.values) max = Math.max(max, Math.abs(value));
    }
    return max || 1;
  }, [mapGrids]);
  const options = useMemo(() => {
    const list: { value: string; label: string }[] = [];
    if (benchmark) list.push({ value: "high_res", label: "高分辨率真值" });
    list.push({ value: "low_res", label: "低分辨率观测（块状显示）" });
    for (const method of methodNames) {
      list.push({ value: `method:${method}`, label: methodLabel(method) });
      if (benchmark) {
        list.push({ value: `difference:${method}`, label: `${methodLabel(method)} − 真值` });
        list.push({ value: `absolute_error:${method}`, label: `${methodLabel(method)} 绝对误差` });
      }
    }
    return list;
  }, [benchmark, methodNames]);
  const overviewRows = useMemo(() => {
    const direction = sortDescending ? -1 : 1;
    const value = (row: RunCase) => overviewMetric === "lr"
      ? row.arrays[overviewSortMethod]?.low_resolution_rmse ?? Number.POSITIVE_INFINITY
      : row.metrics[overviewSortMethod]?.[overviewMetric] ?? Number.POSITIVE_INFINITY;
    return [...runCases].sort((a, b) => direction * (value(a) - value(b)) || a.case_id.localeCompare(b.case_id));
  }, [overviewMetric, overviewSortMethod, runCases, sortDescending]);
  const tradeoffCase = useMemo(() => {
    if (run?.payload.mode !== "benchmark") return null;
    const candidates = runCases.filter((row) => {
      if (row.split !== "test") return false;
      const baseline = row.metrics.bilinear;
      const prior = row.metrics.spectral_prior;
      return baseline && prior && prior.rmse < baseline.rmse && prior.edgeRmse > baseline.edgeRmse;
    });
    const improvement = (row: RunCase) => {
      const baseline = row.metrics.bilinear!;
      const prior = row.metrics.spectral_prior!;
      return (baseline.rmse - prior.rmse) / baseline.rmse;
    };
    return candidates.sort((left, right) => improvement(right) - improvement(left))[0] ?? null;
  }, [run, runCases]);

  const chooseCase = (row: RunCase) => {
    setSelectedSplit(row.split);
    setSelectedCaseId(row.case_id);
    setTab("case");
    setExportArtifactId(null);
  };

  const setPanel = (index: number, value: string) => setPanels((current) => current.map((item, panelIndex) => panelIndex === index ? value : item));
  const fitView = () => setViewport({ zoom: 1, panRows: 0, panCols: 0 });
  const setContextPoint = (which: "A" | "B") => {
    if (contextMenu) selectProfilePoint(which, contextMenu.point);
    setContextMenu(null);
  };

  if (booting) return <div className="boot-screen"><span className="brand-mark">T</span><p>正在恢复本地工作台…</p></div>;

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="TerrainTrace 首页">
          <span className="brand-mark">T</span>
          <span><strong>TerrainTrace</strong><small>地形重建实验室</small></span>
        </a>
        <div className="topbar-status" role="status" aria-live="polite">
          {notice || (dataset ? `${dataset.dataset_id} · revision ${dataset.revision}` : "科学比较工作台 · 合成与用户提供数据")}
        </div>
        <div className="topbar-actions">
          <button className="button quiet" onClick={() => setShowImport(true)}>导入 NPZ</button>
          <button className="button primary" onClick={() => void openDemo()} disabled={busy}>{busy ? "处理中…" : "打开合成示例"}</button>
        </div>
      </header>

      {!dataset && <Welcome onOpenDemo={() => void openDemo()} onImport={() => setShowImport(true)} busy={busy} />}

      {dataset && !run && (
        <section className="dataset-setup page-width">
          <div className="page-heading">
            <div><p className="eyebrow">数据集 / REVISION {dataset.revision}</p><h1>{dataset.dataset_id}</h1></div>
            <span className={`mode-badge ${dataset.mode}`}>{dataset.mode === "benchmark" ? "有 HR 真值 · benchmark" : "LR-only · inference"}</span>
          </div>
          <div className="setup-grid">
            <section className="panel">
              <div className="panel-heading"><div><p className="eyebrow">案例目录</p><h2>{dataset.manifest.cases.length} 个案例</h2></div><span className="muted">一个 revision 内固定案例与 split</span></div>
              <div className="dataset-case-table">
                {dataset.manifest.cases.map((row) => <div className="dataset-case" key={row.id}>
                  <strong>{row.id}</strong><span>{row.group ?? "推断输入"}</span><span>{splitLabel(row.split)}</span><span>{row.shape.join(" × ")} · ×{row.scale}</span>
                </div>)}
              </div>
            </section>
            <section className="panel run-config-card">
              <p className="eyebrow">运行配置</p><h2>选择重建方法</h2>
              <p className="muted">结果绑定当前 revision、输入清单摘要和 terrain_v2 指标协议；同一配置可重跑。</p>
              <div className="method-choices">
                {(["nearest", "bilinear", "spectral_prior"] as const).map((method) => {
                  const disabled = dataset.mode === "inference" && method === "spectral_prior" || dataset.mode === "benchmark" && method === "spectral_prior" && !hasTraining;
                  return <label className={disabled ? "disabled" : ""} key={method}>
                    <input type="checkbox" checked={selectedMethods.includes(method)} disabled={disabled} onChange={() => toggleMethod(method)} />
                    <span><strong>{methodLabel(method)}</strong><small>{methodDescription(method)}</small></span>
                  </label>;
                })}
              </div>
              {dataset.mode === "inference" && <p className="notice compact-notice">没有高分辨率真值，只能比较重建与低分辨率观测的一致性。</p>}
              {dataset.mode === "benchmark" && !hasTraining && <p className="notice compact-notice">当前数据集没有 train 案例，频谱先验不可用。</p>}
              <button className="button primary wide" onClick={() => void startRun()} disabled={busy || !selectedMethods.length}>{busy ? "正在创建任务…" : "运行重建比较"}</button>
            </section>
          </div>
        </section>
      )}

      {dataset && run && (
        <div className="workspace page-width">
          <aside className="case-rail panel">
            <div className="rail-heading"><div><p className="eyebrow">数据集</p><strong>{dataset.dataset_id}</strong></div><span className="revision-chip">r{dataset.revision}</span></div>
            <div className="split-tabs" role="tablist" aria-label="按数据 split 筛选案例">
              {SPLITS.filter((split) => runCases.some((row) => row.split === split.id)).map((split) => {
                const count = runCases.filter((row) => row.split === split.id).length;
                return <button key={split.id} role="tab" aria-selected={selectedSplit === split.id} className={selectedSplit === split.id ? "active" : ""} onClick={() => { setSelectedSplit(split.id); setCaseFilter(""); }}>
                  <span>{split.label}</span><small>{count}</small>
                </button>;
              })}
            </div>
            <label className="case-search"><span className="sr-only">筛选案例</span><input value={caseFilter} placeholder="筛选案例或分组" onChange={(event) => setCaseFilter(event.target.value)} /></label>
            <div className="case-list">
              {filteredCases.map((row) => <button key={row.case_id} className={`case-row ${row.case_id === selectedCaseId ? "selected" : ""}`} onClick={() => chooseCase(row)}>
                <span className="case-status" data-status={row.status} aria-label={row.status} />
                <span><strong>{row.case_id}</strong><small>{row.group ?? "LR-only 输入"}</small></span>
                <span className="case-shape">{row.input_shape[0]}²</span>
              </button>)}
              {filteredCases.length === 0 && <p className="empty-small">当前 split 没有匹配案例。</p>}
            </div>
            <div className="rail-footer"><span>run {run.run_id.slice(-8)}</span><button className="text-button" onClick={() => { setRun(null); setRunCases([]); setSelectedCaseId(null); window.localStorage.setItem(STORAGE_KEY, JSON.stringify({ datasetId: dataset.dataset_id, revision: dataset.revision } satisfies Session)); }}>更改配置</button></div>
          </aside>

          <div className="work-area">
            <div className="work-tabs" role="tablist" aria-label="分析视图">
              <button role="tab" aria-selected={tab === "case"} className={tab === "case" ? "active" : ""} onClick={() => setTab("case")}>案例对照</button>
              <button role="tab" aria-selected={tab === "overview"} className={tab === "overview" ? "active" : ""} onClick={() => setTab("overview")}>全案例指标</button>
              <div className="tab-spacer" />
              <span className={`job-pill ${job?.status ?? run.payload.status}`}><i />{job?.status ?? run.payload.status}{job && !TERMINAL.has(job.status) ? ` · ${job.progress.done}/${job.progress.total}` : ""}</span>
              {job && !TERMINAL.has(job.status) && <button className="button small danger" onClick={() => void cancelRun()}>取消任务</button>}
            </div>

            {job && !TERMINAL.has(job.status) && <div className="progress-card" aria-live="polite">
              <div className="progress-copy"><strong>{job.status === "cancelling" ? "正在结束当前案例" : "重建任务运行中"}</strong><span>进度只按已完成案例计数</span></div>
              <div className="progress-track"><span style={{ width: `${progress}%` }} /></div>
              <b>{job.progress.done}/{job.progress.total}</b>
            </div>}

            {tab === "overview" ? (
              <section className="overview-panel panel">
                <div className="panel-heading"><div><p className="eyebrow">冻结协议 · {run.payload.metrics_protocol}</p><h1>全案例指标矩阵</h1></div><label className="metric-picker">排序指标<select value={overviewMetric} onChange={(event) => setOverviewMetric(event.target.value as OverviewMetric)}>
                  {run.payload.mode === "benchmark" ? <><option value="rmse">高程 RMSE</option><option value="mae">高程 MAE</option><option value="slopeMae">坡度 MAE</option><option value="edgeRmse">边缘 RMSE</option><option value="gradientCorrelation">梯度相关</option></> : null}<option value="lr">LR 一致性</option>
                </select></label><label className="metric-picker">排序方法<select value={overviewSortMethod} onChange={(event) => setOverviewSortMethod(event.target.value)}>{methodNames.map((method) => <option key={method} value={method}>{methodLabel(method)}</option>)}</select></label></div>
                <p className="muted overview-intro">表格保留所有方法的原值；排序只使用你选定的方法与指标，不合成总分。单击案例返回地图对照。</p>
                {runCases.length ? <div className="table-scroll"><table className="metric-matrix"><thead><tr><th><button className="table-sort" onClick={() => setSortDescending((value) => !value)}>案例 / split ↕</button></th>{methodNames.map((method) => <th key={method}>{methodLabel(method)}</th>)}</tr></thead><tbody>
                  {overviewRows.map((row) => <tr key={row.case_id} className="click-row" onClick={() => chooseCase(row)}>
                    <th><strong>{row.case_id}</strong><small>{splitLabel(row.split)} · {row.group ?? "LR-only"}</small></th>
                    {methodNames.map((method) => {
                      const value = overviewMetric === "lr" ? row.arrays[method]?.low_resolution_rmse : row.metrics[method]?.[overviewMetric];
                      return <td key={method} className={value === null || value === undefined ? "na" : ""}>{formatNumber(value, 5)}</td>;
                    })}
                  </tr>)}
                </tbody></table></div> : <EmptyState title="任务尚未产出案例" description="完成的案例会逐项出现在此表。" />}
              </section>
            ) : (
              <div className="case-analysis">
                <section className="viewer-panel panel">
                  <div className="viewer-heading">
                    <div><p className="eyebrow">{caseMeta ? `${splitLabel(activeCase?.split ?? "test")} · ${caseMeta.group ?? "LR-only"} · ${caseMeta.elevation_unit}` : "案例"}</p><h1>{selectedCaseId ?? "等待案例"}</h1></div>
                    <div className="viewer-actions">
                      <button className="icon-button" title="适应视图 (0)" aria-label="适应视图" onClick={fitView}>适应</button>
                      <span className="zoom-readout">{Math.round(viewport.zoom * 100)}%</span>
                      <button className="icon-button" title="导出此案例" onClick={() => void exportCase()} disabled={!activeCase}>导出</button>
                      {exportArtifactId && <a className="download-link" href={`${API}/artifacts/${safeId(exportArtifactId)}/download`}>下载 ZIP</a>}
                    </div>
                  </div>

                  {!activeCase && <EmptyState title="还没有可查看的案例" description="任务运行后，已完成案例会逐项出现在左侧列表。" />}
                  {activeCase && viewer?.loading && <div className="viewer-loading"><span className="spinner" />正在读取服务端数组与误差场…</div>}
                  {activeCase && viewer?.error && <div className="error-card"><strong>数组读取失败</strong><p>{viewer.error}</p><button className="button small" onClick={() => setSelectedCaseId(activeCase.case_id)}>重试</button></div>}
                  {activeCase && viewer && !viewer.loading && !viewer.error && <>
                    <div className="panel-selector-mobile" role="tablist" aria-label="移动端地图选择">
                      {panels.map((key, index) => <button key={`${key}-${index}`} role="tab" aria-selected={mobilePanel === index} className={mobilePanel === index ? "active" : ""} onClick={() => setMobilePanel(index)}>{mapLabel(key)}</button>)}
                    </div>
                    <div className="map-grid">
                      {panels.map((key, index) => {
                        const grid = mapGrids[key];
                        const errorMap = key.startsWith("difference:") || key.startsWith("absolute_error:");
                        const magnitude = key.startsWith("absolute_error:");
                        const range: [number, number] = errorMap ? magnitude ? [0, errorMax] : [-errorMax, errorMax] : elevationRange;
                        return <article className={`map-card ${index === mobilePanel ? "mobile-active" : ""}`} key={`map-${index}`}>
                          <div className="map-card-heading"><div><span className="map-index">0{index + 1}</span><strong>{mapLabel(key)}</strong></div><label><span className="sr-only">地图图层</span><select value={key} onChange={(event) => setPanel(index, event.target.value)}>{options.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}</select></label></div>
                          {grid ? <HeatCanvas
                            grid={grid}
                            label={mapLabel(key)}
                            palette={key.startsWith("difference:") ? "signed" : magnitude ? "magnitude" : "elevation"}
                            range={range}
                            viewport={viewport}
                            setViewport={setViewport}
                            cursor={cursor}
                            setCursor={setCursor}
                            points={profilePoints}
                            roiSelection={roiSelection}
                            placingPoint={placingPoint}
                            setProfilePoint={selectProfilePoint}
                            setRoiSelection={setRoiSelection}
                            finishRoi={(selection) => {
                              const top = Math.max(0, Math.floor(Math.min(selection.start.row, selection.end.row)));
                              const left = Math.max(0, Math.floor(Math.min(selection.start.col, selection.end.col)));
                              const bottom = Math.min(grid.rows, Math.ceil(Math.max(selection.start.row, selection.end.row)) + 1);
                              const right = Math.min(grid.cols, Math.ceil(Math.max(selection.start.col, selection.end.col)) + 1);
                              if (bottom - top > 1 && right - left > 1) void computeRoi([top, left, bottom, right]);
                            }}
                            onContext={(x, y, point) => setContextMenu({ x, y, point })}
                            onEscape={() => { setPlacingPoint(null); setRoiSelection(null); setContextMenu(null); }}
                            onNudge={(rowDelta, colDelta) => nudgeProfile(profileFocus, rowDelta, colDelta)}
                            onMoveCase={(direction) => {
                              const indexInSplit = splitCases.findIndex((row) => row.case_id === selectedCaseId);
                              const next = splitCases[indexInSplit + direction];
                              if (next) chooseCase(next);
                            }}
                            onSetProfileFocus={setProfileFocus}
                          /> : <div className="map-empty">图层数据不存在。请改选当前运行中已生成的方法。</div>}
                          <ColorScale range={range} palette={key.startsWith("difference:") ? "signed" : magnitude ? "magnitude" : "elevation"} unit={errorMap ? magnitude ? caseMeta?.elevation_unit ?? "" : `${caseMeta?.elevation_unit ?? ""} · 重建 − 真值` : caseMeta?.elevation_unit ?? ""} />
                        </article>;
                      })}
                    </div>
                    <div className="viewer-footer"><span>原点左上 · row 向下 · col 向右</span><span>{cursor ? `row ${cursor.row.toFixed(1)} · col ${cursor.col.toFixed(1)}` : "悬停查看网格坐标"}</span><span>滚轮缩放 · 拖动同步平移 · 右键设置剖面端点</span></div>
                    <section className="profile-section">
                      <div className="section-heading"><div><p className="eyebrow">同路径采样</p><h2>高程剖面</h2></div><div className="profile-actions">
                        <button className={`button small ${placingPoint === "A" ? "selected" : ""}`} onClick={() => { setProfileFocus("A"); setPlacingPoint(placingPoint === "A" ? null : "A"); }}>设置 A</button>
                        <button className={`button small ${placingPoint === "B" ? "selected" : ""}`} onClick={() => { setProfileFocus("B"); setPlacingPoint(placingPoint === "B" ? null : "B"); }}>设置 B</button>
                        <button className="button small primary" onClick={() => void sampleProfile()} disabled={!profilePoints.A || !profilePoints.B}>采样剖面</button>
                        {profileData && <button className="button small" onClick={downloadProfile}>下载 CSV</button>}
                      </div></div>
                      <div className="profile-meta">
                        <button className={profileFocus === "A" ? "point-focus" : ""} onClick={() => setProfileFocus("A")} aria-label="选择剖面端点 A">A <span>{formatPoint(profilePoints.A)}</span></button>
                        <div className="nudge-grid" aria-label={`微调端点 ${profileFocus}`}>
                          <button aria-label="端点向上" onClick={() => nudgeProfile(profileFocus, -1, 0)}>↑</button>
                          <button aria-label="端点向左" onClick={() => nudgeProfile(profileFocus, 0, -1)}>←</button>
                          <button aria-label="端点向下" onClick={() => nudgeProfile(profileFocus, 1, 0)}>↓</button>
                          <button aria-label="端点向右" onClick={() => nudgeProfile(profileFocus, 0, 1)}>→</button>
                        </div>
                        <button className={profileFocus === "B" ? "point-focus" : ""} onClick={() => setProfileFocus("B")} aria-label="选择剖面端点 B">B <span>{formatPoint(profilePoints.B)}</span></button>
                      </div>
                      {profileData ? <div ref={chartRef} className="profile-chart" aria-label="服务端返回的多方法共用路径剖面" role="img" /> : <div className="chart-placeholder">设置 A、B 后采样。方法共用同一路径、距离和采样数。</div>}
                    </section>
                  </>}
                </section>
              </div>
            )}
          </div>

          <aside className="detail-rail">
            <section className="panel metric-panel">
              <div className="panel-heading compact-heading"><div><p className="eyebrow">{run.payload.metrics_protocol}</p><h2>案例指标</h2></div><span className="split-chip">{activeCase ? splitLabel(activeCase.split) : "—"}</span></div>
              {activeCase && run.payload.mode === "benchmark" ? <div className="metric-method-list">
                {methodNames.map((method) => {
                  const metric = caseMetrics[method];
                  const array = activeCase.arrays[method];
                  return <details key={method} open={methodNames.length <= 2 || method === "bilinear"}>
                    <summary><strong>{methodLabel(method)}</strong><span>RMSE {formatNumber(metric?.rmse, 5)}</span></summary>
                    <dl className="metric-details">
                      <div><dt>高程 MAE</dt><dd>{formatNumber(metric?.mae)}</dd></div>
                      <div><dt>坡度 MAE</dt><dd>{formatNumber(metric?.slopeMae)}</dd></div>
                      <div><dt>梯度相关</dt><dd>{metric?.gradientCorrelation === null ? "N/A" : formatNumber(metric?.gradientCorrelation)}</dd></div>
                      <div><dt>边缘 RMSE</dt><dd>{formatNumber(metric?.edgeRmse)}</dd></div>
                      <div><dt>LR 一致性</dt><dd>{formatNumber(array?.low_resolution_rmse, 7)}</dd></div>
                      <div><dt>最大面积漂移</dt><dd>{formatNumber(metric?.areaConsistencyMax, 7)}</dd></div>
                    </dl>
                    {metric?.gradientCorrelationReason && <p className="metric-reason">梯度相关未定义：{metric.gradientCorrelationReason}</p>}
                  </details>;
                })}
                <p className="metric-footnote">RMSE、MAE、坡度与边缘以案例真值计算；梯度相关使用 terrain_v2 的二维梯度。</p>
              </div> : activeCase ? <div className="inference-boundary">
                <span className="boundary-icon">LR</span><strong>没有 HR 真值</strong><p>不计算 RMSE、MAE、坡度误差或边缘误差。下方只列重建回投到输入网格后的观测一致性。</p>
                <dl className="metric-details">{methodNames.map((method) => <div key={method}><dt>{methodLabel(method)} · LR RMSE</dt><dd>{formatNumber(activeCase.arrays[method]?.low_resolution_rmse, 7)}</dd></div>)}</dl>
              </div> : <p className="empty-small">选择已完成的案例以查看指标。</p>}
            </section>

            {tradeoffCase && <section className="panel tradeoff-card">
              <p className="eyebrow">TEST 案例 · 结构取舍</p>
              <h2>误差更低，边缘也可能更差</h2>
              <p><strong>{tradeoffCase.case_id}</strong> 中，频谱先验 RMSE 比 bilinear 低 {Math.abs((tradeoffCase.metrics.spectral_prior!.rmse - tradeoffCase.metrics.bilinear!.rmse) / tradeoffCase.metrics.bilinear!.rmse * 100).toFixed(1)}%；边缘 RMSE 高 {((tradeoffCase.metrics.spectral_prior!.edgeRmse - tradeoffCase.metrics.bilinear!.edgeRmse) / tradeoffCase.metrics.bilinear!.edgeRmse * 100).toFixed(1)}%。这是一个合成案例，不代表其他地形或真实数据。</p>
              <button className="button small" onClick={() => chooseCase(tradeoffCase)}>查看该案例与剖面</button>
            </section>}

            {run.payload.prior && <section className="panel prior-panel">
              <div className="panel-heading compact-heading"><div><p className="eyebrow">TRAIN-ONLY · {run.payload.metrics_protocol}</p><h2>频谱先验</h2></div><span className="split-chip">{run.payload.prior.bins} bins</span></div>
              <p className="prior-input">适用输入：{run.payload.prior.compatible_input.shape.join(" × ")} · ×{run.payload.prior.scale} · {run.payload.prior.compatible_input.pixel_size} pixel_size · {run.payload.prior.compatible_input.elevation_unit}</p>
              <PriorGainChart prior={run.payload.prior} />
              <p className="prior-clip">增益截断：{run.payload.prior.clip.map((value) => value.toFixed(2)).join(" – ")} · 虚线为单位增益</p>
              <details className="hash-details"><summary>{run.payload.prior.train_case_hashes.length} 个训练数组 hash</summary><ul>{run.payload.prior.train_case_hashes.map((hash) => <li key={hash} title={hash}>{hash.slice(0, 14)}…</li>)}</ul></details>
            </section>}

            <section className="panel roi-panel">
              <div className="panel-heading compact-heading"><div><p className="eyebrow">局部探索</p><h2>ROI 区域统计</h2></div>{roiData && <span className="exploratory-tag">exploratory</span>}</div>
              <button className={`button wide ${placingPoint === null && roiSelection ? "selected" : ""}`} onClick={() => { setPlacingPoint(null); setRoiSelection({ start: { row: -1, col: -1 }, end: { row: -1, col: -1 } }); setNotice("在地图上拖动框选 ROI；该结果不会替代全案例指标。"); }}>在地图框选 ROI</button>
              {roiData ? <div className="roi-results"><p>row {roiData.bounds[0]}–{roiData.bounds[2]} · col {roiData.bounds[1]}–{roiData.bounds[3]}</p>{Object.entries(roiData.methods).map(([method, stats]) => <div key={method}><strong>{methodLabel(method)}</strong><span>min {formatNumber(stats.min, 3)}</span><span>均值 {formatNumber(stats.mean, 3)}</span><span>max {formatNumber(stats.max, 3)}</span></div>)}<small>局部统计，仅供探索，不替代完整 test 指标。</small></div> : <p className="subtle-copy">框选地图上的一块区域。统计来自服务端原始重建数组。</p>}
            </section>

            <section className="panel run-facts">
              <p className="eyebrow">可复现实验</p><h2>运行来源</h2>
              <dl><div><dt>数据集版本</dt><dd>{run.dataset_id} · r{run.dataset_revision}</dd></div><div><dt>协议</dt><dd>{run.payload.metrics_protocol}</dd></div><div><dt>计算版本</dt><dd>{run.payload.engine_version ?? "—"}</dd></div><div><dt>输入清单 SHA-256</dt><dd title={run.payload.input_sha256}>{run.payload.input_sha256?.slice(0, 18) ?? "—"}…</dd></div><div><dt>配置 SHA-256</dt><dd title={run.payload.config_sha256}>{run.payload.config_sha256?.slice(0, 18) ?? "—"}…</dd></div>
              {run.payload.prior && <div><dt>训练先验</dt><dd>{run.payload.prior.train_case_hashes.length} 个训练组案例 · {run.payload.prior.bins} bins</dd></div>}</dl>
            </section>
          </aside>
        </div>
      )}

      {job?.status === "failed" && <div className="global-error" role="alert"><strong>任务失败</strong><span>{job.error ?? (run?.payload.cases.length ? "查看已有案例；可修正输入后重新运行。" : "请核对方法与输入兼容性后重试。")}</span></div>}
      {showImport && <ImportDialog
        mode={importMode}
        files={files}
        metadata={metadataText}
        busy={busy}
        close={() => setShowImport(false)}
        setMode={switchImportMode}
        setFiles={updateFiles}
        setMetadata={setMetadataText}
        inputRef={fileInput}
        submit={() => void uploadDataset()}
      />}
      {contextMenu && <div className="context-menu" style={{ left: contextMenu.x, top: contextMenu.y }} onPointerDown={(event) => event.stopPropagation()}>
        <p>row {contextMenu.point.row.toFixed(1)} · col {contextMenu.point.col.toFixed(1)}</p>
        <button onClick={() => setContextPoint("A")}>设为剖面 A</button>
        <button onClick={() => setContextPoint("B")}>设为剖面 B</button>
        <button onClick={() => { setProfileFocus("A"); setContextMenu(null); setRoiSelection({ start: contextMenu.point, end: contextMenu.point }); }}>从此处开始 ROI 拖动</button>
      </div>}
      <footer className="app-footer"><span>TerrainTrace 本地实验台</span><span>数据与指标留在本机 · 合成数据不代表真实地形精度</span><button className="text-button" onClick={() => { setDataset(null); setRun(null); setRunCases([]); setJob(null); window.localStorage.removeItem(STORAGE_KEY); setNotice(""); }}>结束本地会话</button></footer>
    </div>
  );
}

function Welcome({ onOpenDemo, onImport, busy }: { onOpenDemo: () => void; onImport: () => void; busy: boolean }) {
  return <main className="welcome page-width">
    <section className="welcome-hero">
      <div className="hero-copy"><p className="eyebrow">SPATIAL RECONSTRUCTION · terrain_v2</p><h1>重建细节，<br /><em>也要守住观测。</em></h1><p>比较高程误差与坡度、边缘结构。每张热图来自真实服务端数组；有真值才计算误差，只有观测时就明确说明能判断什么。</p><div className="hero-actions"><button className="button primary large" onClick={onOpenDemo} disabled={busy}>{busy ? "正在准备…" : "打开固定合成示例"}</button><button className="button large" onClick={onImport}>导入自己的 NPZ</button></div><div className="hero-note"><span className="status-dot" />本地运行 · 不上传到外部服务</div></div>
      <div className="terrain-illustration" aria-hidden="true"><div className="contour contour-one" /><div className="contour contour-two" /><div className="contour contour-three" /><div className="contour-label label-a">ridge</div><div className="contour-label label-b">basin</div><div className="scale-ruler"><span /><span /><span /><small>同一坐标 / 同一色标</small></div></div>
    </section>
    <section className="welcome-principles">
      <article><span className="principle-index">01</span><h2>输入有边界</h2><p>NPZ 只接受受限的二维数组；split、group、单位和间距随案例一起保存。</p></article>
      <article><span className="principle-index">02</span><h2>比较可解释</h2><p>真值、重建与残差共享固定坐标；频谱先验只由训练案例拟合。</p></article>
      <article><span className="principle-index">03</span><h2>没有真值就不打分</h2><p>LR-only 推断只展示观测一致性与局部探索，不伪造 RMSE 或质量排名。</p></article>
    </section>
    <div className="welcome-boundary"><strong>实验范围</strong><span>示例数据是合成地形。真实地理数据还需要坐标系、垂直基准、NoData 与空间切分审查；本版本不会暗示这些检查已完成。</span></div>
  </main>;
}

function ImportDialog({
  mode, files, metadata, busy, close, setMode, setFiles, setMetadata, inputRef, submit,
}: {
  mode: Mode; files: File[]; metadata: string; busy: boolean; close: () => void;
  setMode: (mode: Mode) => void; setFiles: (event: ChangeEvent<HTMLInputElement>) => void;
  setMetadata: (value: string) => void; inputRef: RefObject<HTMLInputElement | null>; submit: () => void;
}) {
  return <div className="modal-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) close(); }}>
    <section className="import-dialog" role="dialog" aria-modal="true" aria-labelledby="import-title">
      <div className="panel-heading"><div><p className="eyebrow">安全导入</p><h2 id="import-title">添加地形案例</h2></div><button className="icon-button close-button" onClick={close} aria-label="关闭导入">×</button></div>
      <p className="muted">单个 NPZ 最大 32 MiB，文件内只允许一个有限二维数组。文件不会离开本机。</p>
      <div className="mode-switch" role="group" aria-label="数据模式"><button className={mode === "benchmark" ? "active" : ""} onClick={() => setMode("benchmark")}>Benchmark · 含 HR</button><button className={mode === "inference" ? "active" : ""} onClick={() => setMode("inference")}>Inference · 仅 LR</button></div>
      <button className="drop-zone" onClick={() => inputRef.current?.click()}><span className="upload-icon">↑</span><strong>{files.length ? `${files.length} 个文件已选` : "选择 .npz 文件"}</strong><small>{mode === "benchmark" ? "每个文件含 high_res 数组" : "仅选一个文件，含 low_res 数组"}</small></button>
      <input ref={inputRef} className="sr-only" type="file" accept=".npz,application/octet-stream" multiple={mode === "benchmark"} onChange={setFiles} />
      {files.length > 0 && <div className="file-list">{files.map((file) => <div key={`${file.name}-${file.size}`}><span>{file.name}</span><small>{(file.size / 1024).toFixed(1)} KB</small></div>)}</div>}
      <label className="metadata-field"><span>案例 metadata JSON <small>{mode === "benchmark" ? "按文件顺序填写 JSON 数组" : "填写单个 JSON 对象"}</small></span><textarea spellCheck={false} value={metadata} onChange={(event) => setMetadata(event.target.value)} placeholder={mode === "benchmark" ? '[{"id":"ridge-01","group":"ridge","split":"train","scale":4,"pixel_size":30,"elevation_unit":"m"}]' : '{"id":"observation-01","scale":4,"pixel_size":30,"elevation_unit":"m"}'} /></label>
      <div className="import-contract"><strong>需要的字段</strong><span>{mode === "benchmark" ? "id · group · split · scale · pixel_size · elevation_unit" : "id · scale · pixel_size · elevation_unit"}</span><small>scale 只能是 2、4 或 8；group 与重复数组不能跨 split。</small></div>
      <div className="dialog-actions"><button className="button" onClick={close} disabled={busy}>取消</button><button className="button primary" onClick={submit} disabled={busy || !files.length}>{busy ? "检查中…" : "检查并导入"}</button></div>
    </section>
  </div>;
}

function HeatCanvas({
  grid, label, palette, range, viewport, setViewport, cursor, setCursor, points, roiSelection, placingPoint,
  setProfilePoint, setRoiSelection, finishRoi, onContext, onEscape, onNudge, onMoveCase, onSetProfileFocus,
}: {
  grid: Grid; label: string; palette: "elevation" | "signed" | "magnitude"; range: [number, number]; viewport: Viewport;
  setViewport: (next: Viewport) => void; cursor: Point | null; setCursor: (point: Point | null) => void;
  points: { A: Point | null; B: Point | null }; roiSelection: { start: Point; end: Point } | null;
  placingPoint: "A" | "B" | null; setProfilePoint: (which: "A" | "B", point: Point) => void;
  setRoiSelection: (selection: { start: Point; end: Point } | null) => void; finishRoi: (selection: { start: Point; end: Point }) => void;
  onContext: (x: number, y: number, point: Point) => void; onEscape: () => void;
  onNudge: (rowDelta: number, colDelta: number) => void; onMoveCase: (direction: -1 | 1) => void;
  onSetProfileFocus: (which: "A" | "B") => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const dragRef = useRef<{ x: number; y: number; viewport: Viewport; mode: "pan" | "roi" | "profile"; start?: Point; end?: Point; profile?: "A" | "B" } | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const draw = () => {
      const rect = canvas.getBoundingClientRect();
      if (!rect.width || !rect.height) return;
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.round(rect.width * dpr));
      canvas.height = Math.max(1, Math.round(rect.height * dpr));
      const context = canvas.getContext("2d");
      if (!context) return;
      context.setTransform(dpr, 0, 0, dpr, 0, 0);
      context.clearRect(0, 0, rect.width, rect.height);
      context.fillStyle = "#f7f9f8";
      context.fillRect(0, 0, rect.width, rect.height);
      const scale = Math.min((rect.width - 34) / grid.cols, (rect.height - 34) / grid.rows) * viewport.zoom;
      const x = rect.width / 2 + (viewport.panCols - grid.cols / 2) * scale;
      const y = rect.height / 2 + (viewport.panRows - grid.rows / 2) * scale;
      const bitmap = document.createElement("canvas");
      bitmap.width = grid.cols;
      bitmap.height = grid.rows;
      const bitmapContext = bitmap.getContext("2d");
      if (!bitmapContext) return;
      const image = bitmapContext.createImageData(grid.cols, grid.rows);
      const stops = palette === "signed" ? ERROR_STOPS : palette === "magnitude" ? ["#f4e2d4", "#eab69e", "#d97c62", "#a94743"] : ELEVATION_STOPS;
      for (let index = 0; index < grid.values.length; index += 1) {
        const value = grid.values[index];
        const t = Math.max(0, Math.min(1, (value - range[0]) / (range[1] - range[0])));
        const [red, green, blue] = interpolateStops(stops, t);
        const offset = index * 4;
        image.data[offset] = red;
        image.data[offset + 1] = green;
        image.data[offset + 2] = blue;
        image.data[offset + 3] = 255;
      }
      bitmapContext.putImageData(image, 0, 0);
      context.imageSmoothingEnabled = false;
      context.drawImage(bitmap, x, y, grid.cols * scale, grid.rows * scale);
      context.strokeStyle = "rgba(32,57,59,.56)";
      context.lineWidth = 1;
      context.strokeRect(x, y, grid.cols * scale, grid.rows * scale);
      const screen = (point: Point) => ({ x: x + (point.col + 0.5) * scale, y: y + (point.row + 0.5) * scale });
      if (points.A && points.B) {
        const first = screen(points.A);
        const second = screen(points.B);
        context.save();
        context.strokeStyle = "#253d3c";
        context.lineWidth = 1.5;
        context.setLineDash([5, 4]);
        context.beginPath();
        context.moveTo(first.x, first.y);
        context.lineTo(second.x, second.y);
        context.stroke();
        context.setLineDash([]);
        for (const [point, color, labelText] of [[first, "#d46a50", "A"], [second, "#306f8d", "B"]] as const) {
          context.beginPath();
          context.fillStyle = "#fff";
          context.strokeStyle = color;
          context.lineWidth = 2;
          context.arc(point.x, point.y, 5.5, 0, Math.PI * 2);
          context.fill();
          context.stroke();
          context.fillStyle = color;
          context.font = "600 10px system-ui";
          context.fillText(labelText, point.x + 8, point.y - 6);
        }
        context.restore();
      }
      if (cursor) {
        const position = screen(cursor);
        context.save();
        context.strokeStyle = "rgba(20,46,48,.72)";
        context.lineWidth = 1;
        context.setLineDash([3, 3]);
        context.beginPath();
        context.moveTo(position.x, y);
        context.lineTo(position.x, y + grid.rows * scale);
        context.moveTo(x, position.y);
        context.lineTo(x + grid.cols * scale, position.y);
        context.stroke();
        context.restore();
      }
      if (roiSelection && roiSelection.start.row >= 0) {
        const top = Math.min(roiSelection.start.row, roiSelection.end.row);
        const left = Math.min(roiSelection.start.col, roiSelection.end.col);
        const bottom = Math.max(roiSelection.start.row, roiSelection.end.row);
        const right = Math.max(roiSelection.start.col, roiSelection.end.col);
        context.fillStyle = "rgba(39,126,121,.18)";
        context.strokeStyle = "#267e79";
        context.lineWidth = 1.5;
        context.fillRect(x + left * scale, y + top * scale, (right - left + 1) * scale, (bottom - top + 1) * scale);
        context.strokeRect(x + left * scale, y + top * scale, (right - left + 1) * scale, (bottom - top + 1) * scale);
      }
    };
    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [cursor, grid, label, palette, points, range, roiSelection, viewport]);

  const pointAt = (event: ReactPointerEvent<HTMLCanvasElement>): Point => {
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    const scale = Math.min((rect.width - 34) / grid.cols, (rect.height - 34) / grid.rows) * viewport.zoom;
    const x = rect.width / 2 + (viewport.panCols - grid.cols / 2) * scale;
    const y = rect.height / 2 + (viewport.panRows - grid.rows / 2) * scale;
    const col = (event.clientX - rect.left - x) / scale - 0.5;
    const row = (event.clientY - rect.top - y) / scale - 0.5;
    return {
      row: Math.max(0, Math.min(grid.rows - 1, row)),
      col: Math.max(0, Math.min(grid.cols - 1, col)),
    };
  };

  const onPointerDown = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    event.currentTarget.setPointerCapture(event.pointerId);
    const point = pointAt(event);
    if (placingPoint) {
      setProfilePoint(placingPoint, { row: Math.round(point.row), col: Math.round(point.col) });
      return;
    }
    const rect = event.currentTarget.getBoundingClientRect();
    const scale = Math.min((rect.width - 34) / grid.cols, (rect.height - 34) / grid.rows) * viewport.zoom;
    const hitRadius = Math.max(0.7, 8 / scale);
    for (const which of ["A", "B"] as const) {
      const endpoint = points[which];
      if (endpoint && Math.hypot(endpoint.row - point.row, endpoint.col - point.col) <= hitRadius) {
        dragRef.current = { x: event.clientX, y: event.clientY, viewport, mode: "profile", profile: which };
        onSetProfileFocus(which);
        return;
      }
    }
    if (roiSelection?.start.row === -1) {
      dragRef.current = { x: event.clientX, y: event.clientY, viewport, mode: "roi", start: point, end: point };
      setRoiSelection({ start: point, end: point });
      return;
    }
    dragRef.current = { x: event.clientX, y: event.clientY, viewport, mode: roiSelection ? "roi" : "pan", start: roiSelection?.start, end: point };
  };

  const onPointerMove = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    const point = pointAt(event);
    setCursor(point);
    const drag = dragRef.current;
    if (!drag) return;
    if (drag.mode === "profile" && drag.profile) {
      setProfilePoint(drag.profile, { row: Math.round(point.row), col: Math.round(point.col) });
      return;
    }
    if (drag.mode === "roi" && drag.start) {
      drag.end = point;
      setRoiSelection({ start: drag.start, end: point });
      return;
    }
    const rect = event.currentTarget.getBoundingClientRect();
    const scale = Math.min((rect.width - 34) / grid.cols, (rect.height - 34) / grid.rows) * drag.viewport.zoom;
    setViewport({ ...drag.viewport, panCols: drag.viewport.panCols + (event.clientX - drag.x) / scale, panRows: drag.viewport.panRows + (event.clientY - drag.y) / scale });
  };

  const onPointerUp = () => {
    const drag = dragRef.current;
    if (drag?.mode === "roi" && drag.start) {
      finishRoi({ start: drag.start, end: drag.end ?? roiSelection?.end ?? drag.start });
      setRoiSelection(null);
    }
    dragRef.current = null;
  };

  const onWheel = (event: ReactWheelEvent<HTMLCanvasElement>) => {
    event.preventDefault();
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    const oldScale = Math.min((rect.width - 34) / grid.cols, (rect.height - 34) / grid.rows) * viewport.zoom;
    const point = pointAt(event as unknown as ReactPointerEvent<HTMLCanvasElement>);
    const nextZoom = Math.max(0.5, Math.min(16, viewport.zoom * (event.deltaY > 0 ? 0.88 : 1.14)));
    const newScale = Math.min((rect.width - 34) / grid.cols, (rect.height - 34) / grid.rows) * nextZoom;
    const pointerX = event.clientX - rect.left - rect.width / 2;
    const pointerY = event.clientY - rect.top - rect.height / 2;
    const panCols = (point.col - grid.cols / 2) + pointerX / newScale;
    const panRows = (point.row - grid.rows / 2) + pointerY / newScale;
    if (oldScale > 0) setViewport({ zoom: nextZoom, panRows, panCols });
  };

  const onKeyDown = (event: ReactKeyboardEvent<HTMLCanvasElement>) => {
    if (event.key === "Escape") { onEscape(); return; }
    if (event.key === "0") { setViewport({ zoom: 1, panRows: 0, panCols: 0 }); return; }
    if (event.key === "[") { onMoveCase(-1); return; }
    if (event.key === "]") { onMoveCase(1); return; }
    if (event.key === "ArrowUp") { event.preventDefault(); onNudge(-1, 0); }
    if (event.key === "ArrowDown") { event.preventDefault(); onNudge(1, 0); }
    if (event.key === "ArrowLeft") { event.preventDefault(); onNudge(0, -1); }
    if (event.key === "ArrowRight") { event.preventDefault(); onNudge(0, 1); }
  };

  const onContextMenu = (event: ReactMouseEvent<HTMLCanvasElement>) => {
    event.preventDefault();
    const point = pointAt(event as unknown as ReactPointerEvent<HTMLCanvasElement>);
    onContext(event.clientX, event.clientY, { row: Math.round(point.row), col: Math.round(point.col) });
  };

  return <canvas
    ref={canvasRef}
    className={`heat-canvas ${placingPoint ? "placing" : ""}`}
    role="application"
    tabIndex={0}
    aria-label={`${label}热图。滚轮缩放，拖动平移；右键可设置剖面端点。`}
    onPointerDown={onPointerDown}
    onPointerMove={onPointerMove}
    onPointerUp={onPointerUp}
    onPointerCancel={onPointerUp}
    onPointerLeave={() => { if (!dragRef.current) setCursor(null); }}
    onWheel={onWheel}
    onKeyDown={onKeyDown}
    onContextMenu={onContextMenu}
  />;
}

function ColorScale({ range, palette, unit }: { range: [number, number]; palette: "elevation" | "signed" | "magnitude"; unit: string }) {
  return <div className="color-scale"><span>{formatNumber(range[0], 2)}</span><div className={`color-ramp ${palette}`} /><span>{formatNumber(range[1], 2)}</span><small>{unit}</small></div>;
}

function PriorGainChart({ prior }: { prior: PriorMeta }) {
  const left = 36;
  const top = 8;
  const width = 268;
  const height = 96;
  const minRadius = prior.radius_centers[0] ?? 0;
  const maxRadius = prior.radius_centers.at(-1) ?? 1;
  const x = (radius: number) => maxRadius === minRadius ? left + width / 2 : left + (radius - minRadius) / (maxRadius - minRadius) * width;
  const y = (gain: number) => top + (1.6 - gain) / 1.1 * height;
  const points = prior.radius_centers.map((radius, index) => `${x(radius)},${y(prior.gains[index] ?? 1)}`).join(" ");
  const ticks = [0.5, 1, 1.5];
  return <svg className="prior-chart-svg" viewBox="0 0 320 138" role="img" aria-label="训练组拟合的径向频谱增益；虚线表示单位增益">
    <title>Train-only radial gain · {prior.bins} bins</title>
    <text className="prior-axis-title" x="7" y="18">增益</text>
    {ticks.map((tick) => <g key={tick}>
      <line className={tick === 1 ? "prior-baseline" : "prior-gridline"} x1={left} x2={left + width} y1={y(tick)} y2={y(tick)} />
      <text className="prior-tick" x={left - 7} y={y(tick) + 3} textAnchor="end">{tick.toFixed(1)}</text>
    </g>)}
    <polyline className="prior-gain-line" points={points} />
    {prior.radius_centers.map((radius, index) => <circle key={`${radius}-${index}`} className="prior-gain-point" cx={x(radius)} cy={y(prior.gains[index] ?? 1)} r="2.8">
      <title>r={radius.toFixed(3)} · gain={(prior.gains[index] ?? 1).toFixed(3)}</title>
    </circle>)}
    <line className="prior-axis" x1={left} x2={left + width} y1={top + height} y2={top + height} />
    <text className="prior-tick" x={left} y={top + height + 13} textAnchor="start">{minRadius.toFixed(2)}</text>
    <text className="prior-tick" x={left + width} y={top + height + 13} textAnchor="end">{maxRadius.toFixed(2)}</text>
    <text className="prior-axis-title" x={left + width / 2} y={top + height + 27} textAnchor="middle">径向频率</text>
  </svg>;
}

function EmptyState({ title, description }: { title: string; description: string }) {
  return <div className="empty-state"><span className="empty-graphic">∿</span><strong>{title}</strong><p>{description}</p></div>;
}

function downloadBlob(blob: Blob, name: string) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  URL.revokeObjectURL(url);
}

function interpolateStops(stops: string[], t: number): [number, number, number] {
  const segment = t * (stops.length - 1);
  const leftIndex = Math.min(stops.length - 2, Math.floor(segment));
  const fraction = segment - leftIndex;
  const left = hexRgb(stops[leftIndex]);
  const right = hexRgb(stops[leftIndex + 1]);
  return [0, 1, 2].map((index) => Math.round(left[index] + (right[index] - left[index]) * fraction)) as [number, number, number];
}

function hexRgb(color: string): [number, number, number] {
  return [1, 3, 5].map((index) => Number.parseInt(color.slice(index, index + 2), 16)) as [number, number, number];
}

function methodLabel(method: string): string {
  return method === "nearest" ? "Nearest" : method === "bilinear" ? "Bilinear" : method === "spectral_prior" ? "频谱先验" : method;
}

function methodDescription(method: string): string {
  return method === "nearest" ? "最近邻单元基线" : method === "bilinear" ? "投影双线性基线" : "仅由 train 案例拟合";
}

function mapLabel(key: string): string {
  if (key === "high_res") return "高分辨率真值";
  if (key === "low_res") return "低分辨率观测";
  if (key.startsWith("difference:")) return `${methodLabel(key.slice(11))} − 真值`;
  if (key.startsWith("absolute_error:")) return `${methodLabel(key.slice(15))} 绝对误差`;
  if (key.startsWith("method:")) return methodLabel(key.slice(7));
  return key;
}

function splitLabel(split: Split): string {
  return SPLITS.find((item) => item.id === split)?.label ?? split;
}

function formatPoint(point: Point | null): string {
  return point ? `row ${point.row.toFixed(0)} · col ${point.col.toFixed(0)}` : "未设置";
}

createRoot(document.getElementById("root")!).render(<App />);
