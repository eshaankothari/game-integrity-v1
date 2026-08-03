import { OrbitControls, Line } from "@react-three/drei";
import { Canvas, ThreeEvent } from "@react-three/fiber";
import { useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import type { Selection } from "../App";
import { CloudNode, fetchCloud } from "../api";

// The anomaly space: every combined_cut game as a node on
// (performance, market, motive), normalized to a unit box. Orbit to rotate,
// hover to identify, click to open the case. Red = tail under 1-in-100;
// dim gray = games the composite run gated out (not clickable).

interface Props {
  dark: boolean;
  onPick: (s: Selection) => void;
}

function normalize(nodes: CloudNode[]) {
  const axes = ["performance", "market", "motive"] as const;
  const lo = axes.map((a) => Math.min(...nodes.map((n) => n[a])));
  const hi = axes.map((a) => Math.max(...nodes.map((n) => n[a])));
  return nodes.map((n) =>
    axes.map((a, i) => ((n[a] - lo[i]) / (hi[i] - lo[i] || 1)) * 1.8 - 0.9),
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
      gated: new THREE.Color(dark ? "#24405a" : "#ccd5de"),
      hot: new THREE.Color(dark ? "#ff6e6c" : "#d8383a"),
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
      const s = i === hovered ? 2.2 : n.tail_pct != null && n.tail_pct < 0.01 ? 1.5 : 1;
      dummy.scale.setScalar(s);
      dummy.updateMatrix();
      m.setMatrixAt(i, dummy.matrix);
      m.setColorAt(
        i,
        i === hovered
          ? palette.hover
          : !n.in_ledger
            ? palette.gated
            : n.tail_pct != null && n.tail_pct < 0.01
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
      <sphereGeometry args={[0.014, 10, 10]} />
      <meshBasicMaterial toneMapped={false} />
    </instancedMesh>
  );
}

function Axes({ dark }: { dark: boolean }) {
  const c = dark ? "#3a5a78" : "#c5ccd4";
  const ends: Array<[[number, number, number], [number, number, number]]> = [
    [[-1, -1, -1], [1.05, -1, -1]],   // performance
    [[-1, -1, -1], [-1, 1.05, -1]],   // market
    [[-1, -1, -1], [-1, -1, 1.05]],   // motive
  ];
  return (
    <>
      {ends.map((pts, i) => (
        <Line key={i} points={pts} color={c} lineWidth={1} />
      ))}
    </>
  );
}

export function Cloud3D({ dark, onPick }: Props) {
  const { data } = useQuery({ queryKey: ["cloud"], queryFn: fetchCloud });
  const [hover, setHover] = useState<CloudNode | null>(null);

  if (!data) return <div className="status">Loading anomaly space…</div>;

  return (
    <div className="cloudwrap">
      <Canvas camera={{ position: [2.1, 1.5, 2.1], fov: 40 }} gl={{ antialias: true }}>
        <Axes dark={dark} />
        <Points nodes={data.nodes} dark={dark} onPick={onPick} onHover={setHover} />
        <OrbitControls enablePan={false} minDistance={1.4} maxDistance={5}
                       autoRotate={hover == null} autoRotateSpeed={0.6} />
      </Canvas>
      <div className="cloud-caption">
        {hover ? (
          <>
            <b>{hover.player}</b> · {hover.game_date} · P {hover.performance.toFixed(2)} ·
            M {hover.market.toFixed(2)} · $ {hover.motive.toFixed(2)}
            {hover.tail_pct != null && (
              <span className={hover.tail_pct < 0.01 ? "hot" : ""}>
                {" "}· 1 in {Math.round(1 / hover.tail_pct).toLocaleString()}
              </span>
            )}
            {!hover.in_ledger && <span> · gated out of ledger</span>}
          </>
        ) : (
          <>drag to rotate · axes: performance / market / motive · red = tail under 1-in-100 · dim = gated out</>
        )}
      </div>
    </div>
  );
}
