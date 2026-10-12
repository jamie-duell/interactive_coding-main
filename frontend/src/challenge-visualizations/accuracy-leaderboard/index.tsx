import { Leaderboard } from '../../components/Leaderboard';
import type { ArenaProps, VisualizationPlugin } from '../types';

function Arena({ challenge, participants, sessionDisplay, scoresHidden, selectedParticipantId, onSelectParticipant, detailPanel }: ArenaProps) {
  const nTest = sessionDisplay?.data.n_test;
  return (
    <div className="arena accuracy-arena">
      <section className="accuracy-ranking" aria-label="Test accuracy leaderboard">
        <h2>{challenge.title}</h2>
        <p className="muted">
          Best test accuracy{typeof nTest === 'number' ? ` on ${nTest.toLocaleString()} test images` : ''}.
          {' '}Select a student to inspect their submissions.
        </p>
        <Leaderboard participants={participants} challenge={challenge} selectedId={selectedParticipantId} onSelect={onSelectParticipant} scoresHidden={scoresHidden} />
        {participants.length === 0 && <p className="muted">Waiting for students to join…</p>}
      </section>
      <section className="accuracy-detail">{detailPanel}</section>
    </div>
  );
}

export const accuracyLeaderboard: VisualizationPlugin = { id: 'accuracy_leaderboard', Arena };
