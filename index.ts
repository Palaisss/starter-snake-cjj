// Welcome to
// __________         __    __  .__                               __
// \______   \_____ _/  |__/  |_|  |   ____   ______ ____ _____  |  | __ ____
//  |    |  _/\__  \\   __\   __\  | _/ __ \ /  ___//    \\__  \ |  |/ // __ \
//  |    |   \ / __ \|  |  |  | |  |_\  ___/ \___ \|   |  \/ __ \|    <\  ___/
//  |________/(______/__|  |__| |____/\_____>______/___|__(______/__|__\\_____>
//
// Battlesnake AI — Territory-first strategy engine
// Core design: BFS diffusion competition, tail control, food zone control,
//              logarithmic length advantage, hazard avoidance, phase-aware weighting.

import runServer from './server';
import { GameState, InfoResponse, MoveResponse, Coord, Board, Battlesnake } from './types';

// ============================================================
// Constants
// ============================================================

const DIR_VECTORS: Record<string, Coord> = {
  up:    { x: 0, y: 1 },
  down:  { x: 0, y: -1 },
  left:  { x: -1, y: 0 },
  right: { x: 1, y: 0 },
};

const DIR_NAMES = ['up', 'down', 'left', 'right'];

const HAZARD_DMG_THRESHOLD = 95;

// ============================================================
// Utility
// ============================================================

function keyOf(c: Coord): string {
  return `${c.x},${c.y}`;
}

function add(a: Coord, b: Coord): Coord {
  return { x: a.x + b.x, y: a.y + b.y };
}

function eq(a: Coord, b: Coord): boolean {
  return a.x === b.x && a.y === b.y;
}

function inBounds(c: Coord, w: number, h: number): boolean {
  return c.x >= 0 && c.x < w && c.y >= 0 && c.y < h;
}

function manhattan(a: Coord, b: Coord): number {
  return Math.abs(a.x - b.x) + Math.abs(a.y - b.y);
}

// ============================================================
// BFS — Flood-fill distance map
// ============================================================

interface FloodResult {
  cells: Set<string>;
  dist: Map<string, number>;
}

function floodFill(
  start: Coord,
  blocked: Set<string>,
  board: Board,
  maxSteps: number,
): FloodResult {
  const startK = keyOf(start);
  const cells = new Set<string>([startK]);
  const dist = new Map<string, number>([[startK, 0]]);
  const q: Coord[] = [start];
  const maxCells = board.width * board.height;

  let i = 0;
  while (i < q.length && cells.size < maxCells) {
    const cur = q[i++];
    const d = dist.get(keyOf(cur))!;
    if (d >= maxSteps) break;

    for (const v of Object.values(DIR_VECTORS)) {
      const n = add(cur, v);
      const nk = keyOf(n);
      if (!inBounds(n, board.width, board.height)) continue;
      if (cells.has(nk)) continue;
      if (blocked.has(nk)) continue;
      cells.add(nk);
      dist.set(nk, d + 1);
      q.push(n);
    }
  }

  return { cells, dist };
}

// ============================================================
// Board snapshot analysis
// ============================================================

interface Analysis {
  hazardSet: Set<string>;
  foodSet: Set<string>;
  foodArr: Coord[];
  mySnake: Battlesnake;
  opponents: Battlesnake[];
  allSnakes: Battlesnake[];
  is1v1: boolean;
  gamePhase: number;      // 0..1
  longestEnemyLen: number;
  hazardDamage: number;   // per-turn hazard damage
}

function analyzeBoard(state: GameState): Analysis {
  const { board, you, turn } = state;
  const all = board.snakes;
  const n = all.length;

  const hazardSet = new Set<string>();
  for (const h of board.hazards) hazardSet.add(keyOf(h));

  const foodSet = new Set<string>();
  for (const f of board.food) foodSet.add(keyOf(f));

  const opponents = all.filter(s => s.id !== you.id);
  const is1v1 = n <= 2;
  const longestEnemyLen = opponents.reduce((m, s) => Math.max(m, s.length), 0);

  // Game phase: driven by turn progress AND remaining snakes
  const estimatedMaxTurns = board.width * board.height * 0.5;
  const phaseByTurn = Math.min(turn / estimatedMaxTurns, 1);
  const phaseBySnakes = Math.max(0, Math.min(1, (8 - n) / 6));
  const gamePhase = is1v1 ? 1.0 : Math.max(phaseByTurn, phaseBySnakes);

  const hazardDamage = board.hazards.length > 0
    ? state.game.ruleset.settings.hazardDamagePerTurn
    : 0;

  return {
    hazardSet,
    foodSet,
    foodArr: board.food,
    mySnake: you,
    opponents,
    allSnakes: all,
    is1v1,
    gamePhase,
    longestEnemyLen,
    hazardDamage,
  };
}

// ============================================================
// Build blocked set for BFS after my hypothetical move
// ============================================================

function buildBlocked(
  myNewHead: Coord,
  eating: boolean,
  analysis: Analysis,
): Set<string> {
  const { mySnake, allSnakes } = analysis;
  const blocked = new Set<string>();

  // My future body
  blocked.add(keyOf(myNewHead));
  const myBody = eating ? mySnake.body : mySnake.body.slice(0, -1);
  for (const seg of myBody) blocked.add(keyOf(seg));

  // Enemy bodies (current) minus their heads & tails (they move too)
  for (const s of allSnakes) {
    if (s.id === mySnake.id) continue;
    for (let i = 1; i < s.body.length - 1; i++) {
      blocked.add(keyOf(s.body[i]));
    }
    // Also block enemy head positions (they stay roughly in same area)
    // but we unblock tails since they'll move
  }

  // My own body already handled above, but also remove my old neck from
  // blocked (the neck is the cell we're moving FROM, it becomes empty)
  blocked.delete(keyOf(mySnake.body[0]));

  return blocked;
}

// ============================================================
// Exclusive territory computation
// ============================================================

function exclusiveTerritory(
  myDist: Map<string, number>,
  enemyDists: Map<string, number>[],
  is1v1: boolean,
  iAmLonger: boolean,
  longerCount: number,
): number {
  let count = 0;
  for (const [cell, md] of myDist) {
    let mine = true;
    for (const ed of enemyDists) {
      const edVal = ed.get(cell);
      if (edVal !== undefined && edVal < md) {
        mine = false;
        break;
      }
      // In 1v1, when distances are equal, the longer snake wins the contested cell
      if (is1v1 && edVal !== undefined && edVal === md && !iAmLonger) {
        mine = false;
        break;
      }
    }
    if (mine) count++;
  }
  return count;
}

// ============================================================
// Per-direction evaluation
// ============================================================

interface EvalScore {
  dir: string;
  safe: boolean;
  total: number;
  territory: number;
  tailCtrl: number;
  foodCtrl: number;
  healthUrgency: number;
  hazardPct: number;
  lengthEdge: number;
}

function evaluateDir(
  dir: string,
  analysis: Analysis,
  state: GameState,
): EvalScore {
  const { board } = state;
  const { mySnake, hazardSet, foodSet, foodArr, opponents, is1v1, gamePhase,
          longestEnemyLen, hazardDamage } = analysis;

  const vec = DIR_VECTORS[dir];
  const newHead = add(mySnake.head, vec);
  const nhk = keyOf(newHead);

  // ---- basic safety ----
  if (!inBounds(newHead, board.width, board.height)) {
    return { dir, safe: false, total: -Infinity, territory: 0, tailCtrl: 0,
             foodCtrl: 0, healthUrgency: 0, hazardPct: 0, lengthEdge: 0 };
  }

  // Check ALL current body cells (all snakes) — this is a hard collision check
  for (const s of analysis.allSnakes) {
    // For enemies, include full body (they may or may not move)
    if (s.id === mySnake.id) {
      // For ourselves, check body except tail (since tail moves)
      const bodyCheck = s.body.slice(0, -1);
      for (const seg of bodyCheck) {
        if (eq(newHead, seg)) {
          return { dir, safe: false, total: -Infinity, territory: 0, tailCtrl: 0,
                   foodCtrl: 0, healthUrgency: 0, hazardPct: 0, lengthEdge: 0 };
        }
      }
    } else {
      for (const seg of s.body) {
        if (eq(newHead, seg)) {
          // Check if this is a tail — might move away
          const isTail = eq(seg, s.body[s.body.length - 1]);
          if (!isTail) {
            return { dir, safe: false, total: -Infinity, territory: 0, tailCtrl: 0,
                     foodCtrl: 0, healthUrgency: 0, hazardPct: 0, lengthEdge: 0 };
          }
          // Tail might still be here if enemy eats food — treat as high risk
        }
      }
    }
  }

  // ---- head-to-head collision check ----
  for (const op of opponents) {
    for (const opDir of DIR_NAMES) {
      const opNewHead = add(op.head, DIR_VECTORS[opDir]);
      if (eq(opNewHead, newHead)) {
        // Potential head-on collision
        if (mySnake.length <= op.length) {
          // We lose or tie → this move is unsafe
          return { dir, safe: false, total: -Infinity, territory: 0, tailCtrl: 0,
                   foodCtrl: 0, healthUrgency: 0, hazardPct: 0, lengthEdge: 0 };
        }
        // We're longer — we win, but still risky if there are multiple enemies
      }
    }
  }

  // ---- high-damage hazard check ----
  // Don't allow moving into >95 damage cells
  const eating = foodSet.has(nhk);
  if (hazardSet.has(nhk) && hazardDamage >= HAZARD_DMG_THRESHOLD) {
    return { dir, safe: false, total: -Infinity, territory: 0, tailCtrl: 0,
             foodCtrl: 0, healthUrgency: 0, hazardPct: 0, lengthEdge: 0 };
  }

  // ---- build blocked set & run BFS ----
  const blocked = buildBlocked(newHead, eating, analysis);

  // Also exclude high-damage hazard cells from BFS entirely
  if (hazardDamage >= HAZARD_DMG_THRESHOLD) {
    for (const hk of hazardSet) blocked.add(hk);
  }

  const myFill = floodFill(newHead, blocked, board, 999);

  const enemyFills: { cells: Set<string>; dist: Map<string, number> }[] = [];
  for (const op of opponents) {
    // Enemy flood fill from their current head (simplified: we don't simulate their move)
    const opBlocked = new Set<string>(blocked);
    // Add their body cells but remove their tail (can't predict exactly)
    // Actually blocked already contains their bodies minus heads/tails from buildBlocked
    // We need to run enemy flood fill WITHOUT my snake's future body in their blocked set
    // since my body is an obstacle for them
    const eFill = floodFill(op.head, opBlocked, board, 999);
    enemyFills.push(eFill);
  }

  // ---- Territory score ----
  const iAmLonger = mySnake.length > longestEnemyLen;
  const longerCount = opponents.filter(o => mySnake.length > o.length).length;
  const exclusive = exclusiveTerritory(myFill.dist, enemyFills.map(e => e.dist), is1v1, iAmLonger, longerCount);
  const totalReachable = myFill.cells.size;
  const territoryScore = exclusive * 1.0 + totalReachable * 0.1;

  // ---- Tail control score ----
  let tailCtrlScore = 0;
  for (const op of opponents) {
    const tailK = keyOf(op.body[op.body.length - 1]);
    if (myFill.cells.has(tailK)) {
      // I can reach their tail
      tailCtrlScore += 3.0;
      // Check if I can reach it before they do
      if (myFill.dist.has(tailK)) {
        const myDist = myFill.dist.get(tailK)!;
        let enemyCloser = false;
        for (const eFill of enemyFills) {
          const eDist = eFill.dist.get(tailK);
          if (eDist !== undefined && eDist < myDist) {
            enemyCloser = true;
            break;
          }
        }
        if (!enemyCloser) tailCtrlScore += 2.0; // I dominate this tail
      }
    }
  }
  // My own tail — if enemies can reach it, that's bad
  const myTailK = keyOf(mySnake.body[mySnake.body.length - 1]);
  for (const eFill of enemyFills) {
    if (eFill.cells.has(myTailK)) {
      tailCtrlScore -= 4.0;
    }
  }

  // ---- Food control score ----
  let foodCtrlScore = 0;
  const health = mySnake.health;
  const boardDiag = Math.max(board.width, board.height);
  const healthUrgency = Math.max(0, (100 - health) / 100); // 0..1

  for (const f of foodArr) {
    const fk = keyOf(f);
    const myDist = myFill.dist.get(fk);
    const belongToMe = myFill.cells.has(fk);

    if (belongToMe) {
      // Food is in my territory
      if (myDist !== undefined) {
        const distFactor = Math.max(0, 1 - myDist / boardDiag);
        foodCtrlScore += 2.0 + distFactor * 3.0;
      } else {
        foodCtrlScore += 2.0;
      }

      // Check if enemies can also reach it
      let contested = false;
      for (const eFill of enemyFills) {
        if (eFill.cells.has(fk)) {
          contested = true;
          break;
        }
      }
      if (!contested) foodCtrlScore += 2.0; // uncontested food = great
    } else {
      // Food might be in enemy territory
      for (const eFill of enemyFills) {
        if (eFill.cells.has(fk)) {
          foodCtrlScore -= 3.0; // enemy controls this food
          break;
        }
      }
    }

    // When health is low, heavily weight food distance
    if (healthUrgency > 0.5 && myDist !== undefined) {
      const urgencyBonus = healthUrgency * 5.0 * Math.max(0, 1 - myDist / boardDiag);
      foodCtrlScore += urgencyBonus;
    }
  }

  // ---- Hazard penalty ----
  let hazardCount = 0;
  for (const c of myFill.cells) {
    if (hazardSet.has(c)) hazardCount++;
  }
  const hazardPct = myFill.cells.size > 0 ? hazardCount / myFill.cells.size : 0;
  const hazardPenalty = hazardPct * 15.0;
  // If damage is low, hazards are manageable
  const hazardWeight = Math.min(hazardDamage / 15, 1);

  // ---- Length edge ----
  const lengthDiff = mySnake.length - longestEnemyLen;
  // Logarithmic: being longer is crucial, but diminishing returns
  const lengthEdge = lengthDiff > 0
    ? Math.log2(1 + lengthDiff) * 3.0
    : lengthDiff * 5.0; // being shorter is very bad

  // ---- Phase-weighted combined score ----
  const w = gamePhase; // 0=early, 1=late

  // Weights shift from length/food → territory/tail as game progresses
  const wTerritory = 1.0 + w * 1.5;
  const wTail = 0.5 + w * 1.5;
  const wFood = 1.0 - w * 0.5;
  const wLength = 1.5 - w * 1.0;
  const wHealth = 1.0 - w * 0.7;

  const total =
    territoryScore * wTerritory +
    tailCtrlScore * wTail +
    foodCtrlScore * wFood +
    lengthEdge * wLength +
    healthUrgency * 5.0 * wHealth -
    hazardPenalty * hazardWeight;

  return {
    dir,
    safe: true,
    total,
    territory: territoryScore,
    tailCtrl: tailCtrlScore,
    foodCtrl: foodCtrlScore,
    healthUrgency,
    hazardPct,
    lengthEdge,
  };
}

// ============================================================
// info / start / end
// ============================================================

function info(): InfoResponse {
  console.log('INFO');
  return {
    apiversion: '1',
    author: 'snake-breeder',
    color: '#3366cc',
    head: 'fang',
    tail: 'curled',
  };
}

function start(_gameState: GameState): void {
  console.log('GAME START');
}

function end(_gameState: GameState): void {
  console.log('GAME OVER');
}

// ============================================================
// move — main entry
// ============================================================

function move(gameState: GameState): MoveResponse {
  const analysis = analyzeBoard(gameState);
  const { mySnake, opponents, is1v1, gamePhase } = analysis;

  // Evaluate all four directions
  const scores = DIR_NAMES.map(dir => evaluateDir(dir, analysis, gameState));

  // Filter safe moves
  const safeMoves = scores.filter(s => s.safe);

  // Logging
  const phaseLabel = is1v1 ? '1v1' : gamePhase < 0.4 ? 'early' : gamePhase < 0.7 ? 'mid' : 'late';
  console.log(`[${gameState.turn}] phase=${phaseLabel}(${gamePhase.toFixed(2)}) alive=${opponents.length+1} health=${mySnake.health} len=${mySnake.length}`);

  if (safeMoves.length === 0) {
    console.log(`  ⚠ No safe moves!`);
    // Last resort: try any direction (might die but better than crashing)
    for (const d of ['up', 'down', 'left', 'right'] as const) {
      const v = DIR_VECTORS[d];
      const nh = add(mySnake.head, v);
      if (inBounds(nh, gameState.board.width, gameState.board.height)) {
        console.log(`  → falling back to ${d}`);
        return { move: d };
      }
    }
    return { move: 'down' };
  }

  // Sort by total score descending
  safeMoves.sort((a, b) => b.total - a.total);
  const chosen = safeMoves[0];

  console.log(`  dirs: ${scores.map(s => `${s.dir}=${s.total.toFixed(1)}${s.safe?'':'❌'}`).join(' ')}`);
  console.log(`  → ${chosen.dir} (terr=${chosen.territory.toFixed(1)} tail=${chosen.tailCtrl.toFixed(1)} food=${chosen.foodCtrl.toFixed(1)} len=${chosen.lengthEdge.toFixed(1)})`);

  return { move: chosen.dir };
}

// ============================================================
// Boot
// ============================================================

runServer({ info, start, move, end });