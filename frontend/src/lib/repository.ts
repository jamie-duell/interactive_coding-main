/** GitHub repository hosting the classroom notebooks and public data. */
export const GITHUB_REPOSITORY = 'jamie-duell/interactive_coding-main';
export const GITHUB_BRANCH = 'main';

export function colabNotebookUrl(challengeId: string): string {
  return colabNotebookFileUrl(`${challengeId}_colab.ipynb`);
}

export function colabNotebookFileUrl(filename: string): string {
  return `https://colab.research.google.com/github/${GITHUB_REPOSITORY}/blob/${GITHUB_BRANCH}/examples/${encodeURIComponent(filename)}`;
}
