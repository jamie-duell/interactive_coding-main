import { decisionBoundaryArena } from './decision-boundary-arena';
import { accuracyLeaderboard } from './accuracy-leaderboard';
import { fallbackVisualization } from './fallback';
import { generalizationArena } from './generalization-arena';
import { deletionArena } from './deletion-arena';
import { generativeArena } from './generative-arena';
import type { VisualizationPlugin } from './types';

const plugins: Record<string, VisualizationPlugin> = {
  [generativeArena.id]: generativeArena,
  [deletionArena.id]: deletionArena,
  [accuracyLeaderboard.id]: accuracyLeaderboard,
  [decisionBoundaryArena.id]: decisionBoundaryArena,
  [generalizationArena.id]: generalizationArena,
};

/** Register an additional plugin at runtime (e.g. from a new challenge module). */
export function registerVisualization(plugin: VisualizationPlugin) {
  plugins[plugin.id] = plugin;
}

export function getVisualization(id: string | null | undefined): VisualizationPlugin {
  return (id && plugins[id]) || fallbackVisualization;
}

export type { ArenaProps, SubmissionPreviewProps, VisualizationPlugin } from './types';
