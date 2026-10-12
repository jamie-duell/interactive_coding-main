import { useEffect, useRef } from 'react';
import { Leaderboard } from '../../components/Leaderboard';
import type { ArenaProps, SubmissionPreviewProps, VisualizationPlugin } from '../types';

interface Example {
  id: number;
  target: number;
  pixels: number[];
  importance: number[];
  curve: number[];
  deletion_auc: number;
}

interface DeletionData {
  method?: string;
  fractions?: number[];
  mean_curve?: number[];
  examples?: Example[];
}

function ImageMap({ pixels, importance, title }: { pixels: number[]; importance?: number[]; title: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const ctx = ref.current?.getContext('2d');
    if (!ctx) return;
    const image = ctx.createImageData(28, 28);
    const max = importance ? Math.max(1e-15, ...importance.map(Math.abs)) : 1;
    for (let i = 0; i < 784; i++) {
      const grey = Math.max(0, Math.min(255, pixels[i] ?? 0));
      const value = importance?.[i] ?? 0;
      const alpha = importance ? Math.min(0.8, Math.abs(value) / max) : 0;
      const color = value >= 0 ? [255, 45, 45] : [45, 90, 255];
      for (let c = 0; c < 3; c++) image.data[4 * i + c] = (1 - alpha) * grey + alpha * color[c];
      image.data[4 * i + 3] = 255;
    }
    ctx.putImageData(image, 0, 0);
  }, [pixels, importance]);
  return <canvas ref={ref} width={28} height={28} className="xai-image" title={title} aria-label={title} role="img" />;
}

function Curve({ fractions, values }: { fractions: number[]; values: number[] }) {
  const left = 44, top = 14, width = 440, height = 170;
  const points = values.map((y, i) => `${left + fractions[i] * width},${top + (1 - y) * height}`).join(' ');
  return (
    <svg className="deletion-curve" viewBox="0 0 520 225" role="img" aria-label="Mean target probability versus fraction of pixels deleted">
      {[0, 0.25, 0.5, 0.75, 1].map((value) => (
        <g key={value}>
          <line x1={left} x2={left + width} y1={top + (1 - value) * height} y2={top + (1 - value) * height} stroke="#e3e6ea" />
          <text x={left - 6} y={top + (1 - value) * height + 4} textAnchor="end">{value.toFixed(2)}</text>
          <text x={left + value * width} y={top + height + 18} textAnchor="middle">{value * 100}%</text>
        </g>
      ))}
      <polyline points={points} fill="none" stroke="#2457c5" strokeWidth={3} />
      <text x={left + width / 2} y={218} textAnchor="middle">pixels deleted →</text>
    </svg>
  );
}

function Explanation({ data, labels, hidden = false }: { data: DeletionData; labels: string[]; hidden?: boolean }) {
  return (
    <div className="xai-explanation">
      <p><strong>{data.method ?? 'Captum attribution'}</strong></p>
      {!hidden && data.fractions && data.mean_curve && (
        <>
          <h3>Mean deletion curve</h3>
          <p className="muted small">Original target probability as high-ranked pixels are removed. Lower area is better.</p>
          <Curve fractions={data.fractions} values={data.mean_curve} />
        </>
      )}
      <div className="xai-examples">
        {(data.examples ?? []).map((example) => (
          <figure key={example.id}>
            <div className="xai-image-pair">
              <ImageMap pixels={example.pixels} title={`Original image ${example.id}`} />
              <ImageMap pixels={example.pixels} importance={example.importance} title={`Attribution heatmap ${example.id}`} />
            </div>
            <figcaption>
              #{example.id} · explains {labels[example.target] ?? example.target}
              {!hidden && <span> · deletion AUC {example.deletion_auc.toFixed(4)}</span>}
            </figcaption>
          </figure>
        ))}
      </div>
      <p className="muted small">Original → attribution overlay. Red: positive importance; blue: negative.</p>
    </div>
  );
}

function Arena({ challenge, participants, sessionDisplay, scoresHidden, selectedParticipantId, onSelectParticipant, selected, detailPanel }: ArenaProps) {
  const labels = (sessionDisplay?.data.class_labels as string[] | undefined) ?? ['normal', 'pneumonia'];
  return (
    <div className="arena accuracy-arena">
      <section className="accuracy-ranking">
        <h2>{challenge.title}</h2>
        <p className="muted">Find the Captum method with the lowest mean deletion AUC. Lower scores rank first.</p>
        <Leaderboard participants={participants} challenge={challenge} selectedId={selectedParticipantId} onSelect={onSelectParticipant} scoresHidden={scoresHidden} />
        {selected.display ? (
          <Explanation data={selected.display.data as DeletionData} labels={labels} hidden={scoresHidden} />
        ) : <p className="muted">Select an evaluated submission to inspect its curve and attribution maps.</p>}
      </section>
      <section className="accuracy-detail">{detailPanel}</section>
    </div>
  );
}

function SubmissionPreview({ sessionDisplay, display, scoresHidden = false }: SubmissionPreviewProps) {
  return <Explanation data={display.data as DeletionData} labels={(sessionDisplay.data.class_labels as string[]) ?? ['normal', 'pneumonia']} hidden={scoresHidden} />;
}

export const deletionArena: VisualizationPlugin = { id: 'deletion_arena', Arena, SubmissionPreview };
