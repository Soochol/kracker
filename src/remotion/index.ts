// Remotion 모듈 exports
import { registerRoot } from "remotion";
import { RemotionRoot } from "./Root";

// Remotion Entry Point
registerRoot(RemotionRoot);

export { RemotionRoot } from "./Root";
export { VideoComposition } from "./VideoComposition";
export { KenBurnsScene } from "./components/KenBurnsScene";
export { SubtitleOverlay } from "./components/SubtitleOverlay";
export * from "./types";
