import { OrbitControls, Line, Text } from "@react-three/drei";
import { Canvas, ThreeEvent } from "@react-three/fiber";
import { useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import type { Selection } from "../App";
import { CloudNode, fetchCloud } from "../api";
import { isConfirmed } from "../severity";

// The anomaly space: EVERY propped player-game (all 15,494) as a node on
// (performance, market, motive). The 4,810 that survive the cuts are lit; the
// 10,684 the cuts removed are dim and not clickable.
//
// Showing the eliminated games is the point. With only the shortlist the cloud
// is a shape with no reference -- you cannot see that the survivors occupy one
// corner of the space rather than being scattered through it, and you cannot
// see how close a cut game came to surviving.

interface Props {
  dark: boolean;
  onPick: (s: Selection) => void;
}

const AXES = ["performance", "market", "motive"] as const;

// Axis ends are fixed to the 0-100 scale rather than the observed min/max, so
// the box means the same thing every run and a node's position is readable as
// a score. Math.min(...array) is avoided deliberately: it spreads every element
// as an argument, which throws on large arrays -- and this now takes 15,494.
function normalize(nodes: CloudNode[]) {
  return nodes.map((n) => AXES.map((a) => (n[a] / 100) * 1.8 - 0.9));
}

// One tick label per axis end plus the axis name, drawn in 3D so they rotate
// with the box and never disagree with what they are labelling.
function AxisLabel({ pos, text, color, size = 0.075 }: {
  pos: [number, number, number];
  text: string;
  color: string;
  size?: number;
}) {
  return (
    <Text position={pos} fontSize={size} color={color}
          anchorX="center" anchorY="middle">
      {text}
    </Text>
  );
}

function Axes({ dark }: { dark: boolean }) {
  const line = dark ? "#3a5a78" : "#c5ccd4";
  const label = dark ? "#8fa8bd" : "#5a6572";
  const strong = dark ? "#cfe0ee" : "#051c2c";
  const O = -0.95;                       // origin corner
  const E = 0.95;                        // far end of each axis
  const ends: Array<[[number, number, number], [number, number, number]]> = [
    [[O, O, O], [E, O, O]],
    [[O, O, O], [O, E, O]],
    [[O, O, O], [O, O, E]],
  ];
  return (
    <>
      {ends.map((pts, i) => (
        <Line key={i} points={pts} color={line} lineWidth={1.2} />
      ))}

      {/* axis names, out past the far end so they never sit on the data */}
      <AxisLabel pos={[E + 0.22, O, O]} text="PERFORMANCE" color={strong} size={0.085} />
      <AxisLabel pos={[O, E + 0.18, O]} text="MARKET" color={strong} size={0.085} />
      <AxisLabel pos={[O, O, E + 0.2]} text="MOTIVE" color={strong} size={0.085} />

      {/* what each axis MEANS, one line under the name */}
      <AxisLabel pos={[E + 0.22, O - 0.12, O]} text="  worse night ->" color={label} size={0.05} />
      <AxisLabel pos={[O, E + 0.08, O]} text="more under leaning ->" color={label} size={0.05} />
      <AxisLabel pos={[O, O - 0.1, E + 0.2]} text="lower paid ->" color={label} size={0.05} />

      {/* 0 and 100 ticks, so a position reads as a score */}
      <AxisLabel pos={[O - 0.1, O - 0.1, O]} text="0" color={label} size={0.055} />
      <AxisLabel pos={[E, O - 0.12, O]} text="100" color={label} size={0.055} />
      <AxisLabel pos={[O - 0.12, E, O]} text="100" color={label} size={0.055} />
      <AxisLabel pos={[O - 0.1, O - 0.1, E]} text="100" color={label} size={0.055} />
    </>
  );
}

function Points({ nodes, dark, onPick, onHover }: {
  nodes: CloudNode[];
  dark: boolean;
  onPick: (s: Selection) => void;
  onHover: (n: CloudNode | null) => void;
}) {
  const mesh = useRef<THREE.InstancedMesh>(null);
  const positions = useMemo(() => normalize(nodes), [nodes]);
  const [hovered, setHovered] = useState<number | null>(null);

  const palette = useMemo(
    () => ({
      base: new THREE.Color(dark ? "#5b7cff" : "#2251ff"),
      gated: new THREE.Color(dark ? "#1b2c3c" : "#dde3e9"),
      hot: new THREE.Color(dark ? "#ff6e6c" : "#d8383a"),
      conf: new THREE.Color(dark ? "#c8514f" : "#8f1e21"),
      hover: new THREE.Color(dark ? "#ffffff" : "#051c2c"),
    }),
    [dark],
  );

  useEffect(() => {
    const m = mesh.current;
    if (!m) return;
    const dummy = new THREE.Object3D();
    nodes.forEach((n, i) => {
      dummy.position.set(positions[i][0], positions[i][1], positions[i][2]);
      // Eliminated games are drawn smaller as well as dimmer. At 15,494 nodes
      // colour alone is not enough separation -- the gated majority would still
      // read as a wall of dots behind the shortlist.
      const conf = isConfirmed(n.player_id, n.game_id);
      const hot = n.tail_pct != null && n.tail_pct < 0.01;
      const s =
        i === hovered ? 2.4 : conf ? 1.9 : !n.in_ledger ? 0.55 : hot ? 1.5 : 1;
      dummy.scale.setScalar(s);
      dummy.updateMatrix();
      m.setMatrixAt(i, dummy.matrix);
      m.setColorAt(
        i,
        i === hovered
          ? palette.hover
          : conf
            ? palette.conf // ground truth outranks every score color
            : !n.in_ledger
              ? palette.gated
              : hot
                ? palette.hot
                : palette.base,
      );
    });
    m.instanceMatrix.needsUpdate = true;
    if (m.instanceColor) m.instanceColor.needsUpdate = true;
  }, [nodes, positions, palette, hovered]);

  const over = (e: ThreeEvent<PointerEvent>) => {
    e.stopPropagation();
    const id = e.instanceId ?? null;
    setHovered(id);
    onHover(id == null ? null : nodes[id]);
    document.body.style.cursor =
      id != null && nodes[id].in_ledger ? "pointer" : "default";
  };

  return (
    <instancedMesh
      ref={mesh}
      args={[undefined, undefined, nodes.length]}
      onPointerMove={over}
      onPointerOut={() => {
        setHovered(null);
        onHover(null);
        document.body.style.cursor = "default";
      }}
      onClick={(e) => {
        e.stopPropagation();
        const n = e.instanceId != null ? nodes[e.instanceId] : null;
        if (n?.in_ledger) onPick({ playerId: n.player_id, gameId: n.game_id });
      }}
    >
      <sphereGeometry args={[0.012, 8, 8]} />
      <meshBasicMaterial toneMapped={false} />
    </instancedMesh>
  );
}

export function Cloud3D({ dark, onPick }: Props) {
  const { data } = useQuery({ queryKey: ["cloud"], queryFn: fetchCloud });
  const [hover, setHover] = useState<CloudNode | null>(null);
  const [showGated, setShowGated] = useState(true);

  const nodes = useMemo(
    () => (data ? (showGated ? data.nodes : data.nodes.filter((n) => n.in_ledger)) : []),
    [data, showGated],
  );

  if (!data) return <div className="status">Loading anomaly space…</div>;

  return (
    <div className="cloudwrap">
      {/* camera pulled back + slightly wider fov so the axis names and tick
          labels (which sit past the box ends) are inside the frame on load */}
      <Canvas camera={{ position: [2.7, 2.0, 2.7], fov: 42 }} gl={{ antialias: true }}>
        <Axes dark={dark} />
        <Points nodes={nodes} dark={dark} onPick={onPick} onHover={setHover} />
        <OrbitControls enablePan={false} minDistance={1.4} maxDistance={5}
                       autoRotate={hover == null} autoRotateSpeed={0.6} />
      </Canvas>

      <button className="cloud-toggle" onClick={() => setShowGated((v) => !v)}>
        {showGated ? `hide the ${(data.nodes.length - data.shortlist).toLocaleString()} cut games`
                   : `show all ${data.nodes.length.toLocaleString()} propped games`}
      </button>

      <div className="cloud-caption">
        {hover ? (
          <>
            <b>{hover.player}</b> · {hover.game_date}
            {hover.points != null && hover.line != null && (
              <> · {hover.points} pts on {hover.line}</>
            )}
            {" · "}performance {hover.performance.toFixed(1)} · market{" "}
            {hover.market.toFixed(1)} · motive {hover.motive.toFixed(1)}
            {hover.in_ledger
              ? <b> · rank #{hover.rank?.toLocaleString()}</b>
              : <span> · cut by {data.cuts[String(hover.cut)] ?? "a cut"}</span>}
          </>
        ) : (
          <>
            {nodes.length.toLocaleString()} propped games · {data.shortlist.toLocaleString()} on
            the shortlist (blue), the rest cut (faint) · red = most suspcious games ·  each axis 0–100, higher = more
            suspicious
          </>
        )}
      </div>
    </div>
  );
}
