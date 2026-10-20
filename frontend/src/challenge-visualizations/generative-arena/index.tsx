import { useState } from 'react';
import { api } from '../../api/client';
import { useLiveSession } from '../../api/live';
import type { ChallengeSummary, ParticipantLive, SubmissionDisplayOut } from '../../api/types';
import { Leaderboard } from '../../components/Leaderboard';
import type { ArenaProps, SubmissionPreviewProps, VisualizationPlugin } from '../types';

function Gallery({ display }: { display: SubmissionDisplayOut }) {
  const data = display.data as { model_family?: string; description?: string; samples?: number[][] };
  return <section className="generated-gallery">
    <h3>{data.model_family === 'vae' ? 'VAE' : 'Diffusion'} generated samples</h3>
    <p>{data.description}</p>
    <div className="generated-grid">{(data.samples ?? []).map((pixels, i) => (
      <svg key={i} viewBox="0 0 28 28" shapeRendering="crispEdges" role="img" aria-label={`Generated sample ${i + 1}`}>
        {pixels.map((v, j) => <rect key={j} x={j % 28} y={Math.floor(j / 28)} width={1} height={1} fill={`rgb(${v},${v},${v})`} />)}
      </svg>
    ))}</div>
  </section>;
}

interface BoardsProps {
  participants: ParticipantLive[];
  challenge: ChallengeSummary;
  scoresHidden: boolean;
  onSelect?: (participantId: string, submissionId: string) => void;
}

export function GenerativeLeaderboards({ participants, challenge, scoresHidden, onSelect }: BoardsProps) {
  const [preview, setPreview] = useState<SubmissionDisplayOut | null>(null);
  const [error, setError] = useState('');
  const [chosen, setChosen] = useState<{ id: string; family: string } | null>(null);
  const families = Object.entries(challenge.leaderboard_tracks ?? {});
  return <>
    <div className="generative-boards" aria-label="VAE and diffusion leaderboards">
      {families.map(([family, title]) => {
        const rows = participants.map((p) => p.tracks?.[family] ?? {
          ...p, best_score: null, best_metrics: null, best_submission_id: null, rank: null, n_submissions: 0, latest_status: null, latest_at: null,
        });
        return <section key={family} className="generative-board">
          <h3>{title} leaderboard</h3>
          <p className="muted small">Your best {family === 'vae' ? 'VAE' : 'diffusion'} score · lower feature MMD² is better</p>
          <div className="generative-table"><Leaderboard participants={rows} challenge={challenge} compact
            selectedId={chosen?.family === family ? chosen.id : null} scoresHidden={scoresHidden}
            onSelect={(id) => {
              const row = rows.find((p) => p.id === id);
              if (!row?.best_submission_id) return;
              setChosen({ id: row.id, family });
              if (onSelect) { onSelect(row.id, row.best_submission_id); return; }
              setError('');
              api.submissionDisplay(row.best_submission_id).then(setPreview).catch(() => setError('Could not load generated samples.'));
            }} /></div>
        </section>;
      })}
    </div>
    {error && <p role="alert" className="error">{error}</p>}
    {preview && !onSelect && <Gallery display={preview} />}
  </>;
}

export function GenerativeSessionPanel({ sessionId, teacher = false }: { sessionId: string; teacher?: boolean }) {
  const { state, error } = useLiveSession(sessionId);
  if (error) return <p className="error">{error}</p>;
  if (state.deleted) return <p className="muted">Session deleted.</p>;
  if (!state.challenge || !state.session) return <p className="muted">Loading leaderboards…</p>;
  return <GenerativeLeaderboards participants={state.participants} challenge={state.challenge} scoresHidden={!teacher && state.session.scores_hidden} />;
}

function Arena({ challenge, participants, scoresHidden, onSelectParticipant, selected, detailPanel }: ArenaProps) {
  return <div className="generative-arena">
    <section><h2>{challenge.title}</h2>
      <GenerativeLeaderboards participants={participants} challenge={challenge} scoresHidden={scoresHidden}
        onSelect={(id, submissionId) => onSelectParticipant(id, submissionId)} />
      {selected.display && <Gallery display={selected.display} />}
    </section>
    <section className="generative-detail">{detailPanel}</section>
  </div>;
}

function SubmissionPreview({ display }: SubmissionPreviewProps) { return <Gallery display={display} />; }

export const generativeArena: VisualizationPlugin = { id: 'generative_arena', Arena, SubmissionPreview };
