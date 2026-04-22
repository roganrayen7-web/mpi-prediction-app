import React, { useState, useMemo, useEffect } from 'react';
import { 
  LayoutDashboard, 
  Map as MapIcon, 
  Settings2, 
  BarChart3, 
  Zap, 
  ChevronRight,
  TrendingDown,
  TrendingUp,
  AlertTriangle,
  CheckCircle2,
  Filter,
  Info,
  ArrowRight
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  ResponsiveContainer, 
  ScatterChart, 
  Scatter, 
  XAxis, 
  YAxis, 
  ZAxis, 
  Tooltip, 
  Cell, 
  BarChart, 
  Bar, 
  CartesianGrid,
  Legend,
  ReferenceArea
} from 'recharts';
import { 
  generateMockData, 
  District, 
  STATES, 
  predictStagnationRisk, 
  getRiskBand, 
  FEATURE_IMPORTANCE,
  DistrictIndicators
} from './data';
import { cn } from './lib/utils';

// --- Sub-components ---

const Badge = ({ children, variant = 'default' }: { children: React.ReactNode, variant?: 'default' | 'success' | 'warning' | 'error' }) => {
  const variants = {
    default: 'bg-slate-100 text-slate-700 border-slate-200',
    success: 'bg-emerald-50 text-emerald-700 border-emerald-100',
    warning: 'bg-amber-50 text-amber-700 border-amber-100',
    error: 'bg-rose-50 text-rose-700 border-rose-100',
  };
  return (
    <span className={cn("px-2 py-0.5 rounded-full text-xs font-medium border uppercase tracking-wider", variants[variant])}>
      {children}
    </span>
  );
};

const StatCard = ({ title, value, subValue, icon: Icon, delta, deltaType }: { 
  title: string, value: string | number, subValue?: string, icon: any, delta?: string, deltaType?: 'positive' | 'negative' | 'neutral'
}) => (
  <motion.div 
    initial={{ opacity: 0, y: 10 }}
    animate={{ opacity: 1, y: 0 }}
    className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden"
  >
    <div className="p-4 flex justify-between items-start">
      <div>
        <p className="text-[10px] uppercase font-bold text-slate-400 tracking-wider mb-1">{title}</p>
        <h3 className="text-2xl font-bold tracking-tight text-slate-800">{value}</h3>
        {subValue && <p className="text-xs text-slate-500 mt-1">{subValue}</p>}
      </div>
      <div className="p-2 bg-slate-50 border border-slate-100 rounded-lg shrink-0">
        <Icon className="w-5 h-5 text-indigo-600" />
      </div>
    </div>
    {delta && (
      <div className={cn(
        "px-4 py-1.5 text-xs font-semibold flex items-center gap-1 border-t border-slate-50",
        deltaType === 'positive' ? "bg-emerald-50 text-emerald-700" : 
        deltaType === 'negative' ? "bg-rose-50 text-rose-700" : "bg-slate-50 text-slate-700"
      )}>
        {deltaType === 'positive' ? <TrendingDown className="w-3 h-3" /> : <TrendingUp className="w-3 h-3" />}
        {delta}
      </div>
    )}
  </motion.div>
);

const Slider = ({ label, value, onChange, min = 0, max = 30 }: { label: string, value: number, onChange: (v: number) => void, min?: number, max?: number }) => (
  <div className="space-y-1.5">
    <div className="flex justify-between text-xs text-slate-500">
      <span className="font-medium">{label}</span>
      <span className="font-bold text-indigo-600">+{value}%</span>
    </div>
    <div className="relative flex items-center">
      <input 
        type="range" 
        min={min} 
        max={max} 
        value={value} 
        onChange={(e) => onChange(parseInt(e.target.value))}
        className="w-full h-1.5 bg-slate-100 rounded-full appearance-none cursor-pointer accent-indigo-500"
      />
    </div>
  </div>
);

// --- Main App ---

export default function App() {
  const [activeTab, setActiveTab] = useState('hotspot');
  const [allDistricts] = useState(() => generateMockData());
  const [selectedState, setSelectedState] = useState('All');
  const [selectedDistrictId, setSelectedDistrictId] = useState<string | null>(null);
  
  // Policy sliders
  const [sliders, setSliders] = useState({
    sanitation: 5,
    fuel: 5,
    schooling: 5,
    housing: 5,
    bank: 5,
    health: 5
  });

  const filteredDistricts = useMemo(() => {
    let list = allDistricts.map(d => ({
      ...d,
      stagnation_risk_prob: predictStagnationRisk(d, d.mpi_previous),
      risk_band: getRiskBand(predictStagnationRisk(d, d.mpi_previous))
    }));

    if (selectedState !== 'All') {
      list = list.filter(d => d.state === selectedState);
    }
    return list;
  }, [allDistricts, selectedState]);

  useEffect(() => {
    if (filteredDistricts.length > 0 && (!selectedDistrictId || !filteredDistricts.find(d => d.id === selectedDistrictId))) {
      setSelectedDistrictId(filteredDistricts[0].id);
    }
  }, [filteredDistricts, selectedDistrictId]);

  const currentDistrict = useMemo(() => {
    return filteredDistricts.find(d => d.id === selectedDistrictId) || null;
  }, [filteredDistricts, selectedDistrictId]);

  const scenarioResult = useMemo(() => {
    if (!currentDistrict) return null;

    const indicators: DistrictIndicators = {
      sanitation_pct: Math.min(100, currentDistrict.sanitation_pct + sliders.sanitation),
      clean_fuel_pct: Math.min(100, currentDistrict.clean_fuel_pct + sliders.fuel),
      schooling_years_pct: Math.min(100, currentDistrict.schooling_years_pct + sliders.schooling),
      housing_pct: Math.min(100, currentDistrict.housing_pct + sliders.housing),
      bank_access_pct: Math.min(100, currentDistrict.bank_access_pct + sliders.bank),
      female_literacy_pct: currentDistrict.female_literacy_pct,
      rural_pop_pct: currentDistrict.rural_pop_pct,
      health_access_pct: Math.min(100, currentDistrict.health_access_pct + sliders.health),
    };

    const baseProb = predictStagnationRisk(currentDistrict, currentDistrict.mpi_previous);
    const scenarioProb = predictStagnationRisk(indicators, currentDistrict.mpi_previous);
    
    const improvementScore = (
      sliders.sanitation * 0.15 +
      sliders.fuel * 0.18 +
      sliders.schooling * 0.20 +
      sliders.housing * 0.17 +
      sliders.bank * 0.12 +
      sliders.health * 0.18
    ) / 100;

    const projectedMpi = Math.max(0, currentDistrict.mpi_current * (1 - improvementScore));

    return {
      indicators,
      baseProb,
      scenarioProb,
      projectedMpi
    };
  }, [currentDistrict, sliders]);

  const avgMpi = filteredDistricts.reduce((acc, d) => acc + d.mpi_current, 0) / filteredDistricts.length;
  const highRiskCount = filteredDistricts.filter(d => d.risk_band === 'High').length;

  return (
    <div className="flex h-screen bg-slate-50 font-sans text-slate-900 overflow-hidden">
      {/* Sidebar - Policy Controls */}
      <aside className="w-[280px] bg-white border-r border-slate-200 flex flex-col flex-shrink-0 z-10 shadow-sm overflow-y-auto">
        <div className="p-5 border-b border-slate-200 bg-slate-50/50">
          <div className="flex items-center gap-2.5 mb-2">
            <div className="w-9 h-9 rounded-lg bg-indigo-600 shadow-lg shadow-indigo-200 flex items-center justify-center text-white">
              <Zap className="w-5 h-5" />
            </div>
            <div>
              <h1 className="text-base font-bold text-slate-900 tracking-tight leading-none mb-0.5">Unity Simulator</h1>
              <span className="text-[10px] font-bold text-indigo-600 tracking-wider uppercase">Pro Engine</span>
            </div>
          </div>
        </div>

        <div className="flex-1 p-5 space-y-6">
          <div className="space-y-4">
            <h2 className="text-[11px] font-bold text-slate-500 uppercase tracking-wider">Region Filters</h2>
            
            <div className="space-y-2">
              <label className="text-[11px] font-semibold text-slate-500 uppercase tracking-tight">Select State</label>
              <select 
                className="h-9 w-full rounded border border-slate-300 bg-white px-3 flex items-center text-sm shadow-sm focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 outline-none transition-all"
                value={selectedState}
                onChange={(e) => setSelectedState(e.target.value)}
              >
                <option value="All">National View</option>
                {STATES.map(s => <option key={s} value={s}>{s}</option>)}
              </select>
            </div>

            <div className="space-y-2">
              <label className="text-[11px] font-semibold text-slate-500 uppercase tracking-tight">Select District</label>
              <select 
                className="h-9 w-full rounded border border-slate-300 bg-white px-3 flex items-center text-sm shadow-sm focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 outline-none transition-all"
                value={selectedDistrictId || ''}
                onChange={(e) => setSelectedDistrictId(e.target.value)}
              >
                {filteredDistricts.map(d => <option key={d.id} value={d.id}>{d.district}</option>)}
              </select>
            </div>
          </div>

          <div className="pt-4 border-t border-slate-100">
            <h3 className="text-xs font-bold text-slate-700 mb-4">Intervention Scenario (%)</h3>
            <div className="space-y-4">
              <Slider label="Sanitation Expansion" value={sliders.sanitation} onChange={(v) => setSliders(s => ({...s, sanitation: v}))} />
              <Slider label="Clean Fuel Adoption" value={sliders.fuel} onChange={(v) => setSliders(s => ({...s, fuel: v}))} />
              <Slider label="Schooling Improvement" value={sliders.schooling} onChange={(v) => setSliders(s => ({...s, schooling: v}))} />
              <Slider label="Housing Upgrades" value={sliders.housing} onChange={(v) => setSliders(s => ({...s, housing: v}))} />
              <Slider label="Financial Inclusion" value={sliders.bank} onChange={(v) => setSliders(s => ({...s, bank: v}))} />
              <Slider label="Health Access" value={sliders.health} onChange={(v) => setSliders(s => ({...s, health: v}))} />
            </div>
          </div>
        </div>

        <div className="p-5 bg-slate-50 border-t border-slate-200">
          <button className="w-full bg-indigo-600 text-white py-2.5 rounded-lg text-sm font-semibold shadow-md hover:bg-indigo-700 transition-all flex items-center justify-center gap-2 group">
            Apply Simulations
            <Zap className="p-0 w-3.5 h-3.5 scale-750 opacity-50 group-hover:opacity-100 transition-opacity" />
          </button>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col h-full overflow-hidden">
        {/* Header/Tool bar */}
        <header className="bg-white border-b border-slate-200 px-8 py-4 flex justify-between items-center z-10 shrink-0">
          <div>
            <h1 className="text-xl font-bold text-slate-900 tracking-tight leading-none mb-1">Unity: MPI Simulator</h1>
            <p className="text-sm text-slate-500 font-medium">Predicting multi-dimensional poverty trajectories for {currentDistrict?.district}, {currentDistrict?.state}</p>
          </div>
          <div className="flex items-center gap-4">
            {currentDistrict?.risk_band === 'High' && (
              <span className="text-[10px] font-bold px-2.5 py-1 rounded bg-rose-100 text-rose-700 border border-rose-200 uppercase tracking-wider">High Risk District</span>
            )}
            <div className="h-10 w-10 rounded-full bg-slate-50 border border-slate-200 flex items-center justify-center text-lg text-slate-400">📊</div>
          </div>
        </header>

        {/* Tab Bar */}
        <div className="bg-white border-b border-slate-200 flex px-8 gap-8 shrink-0">
          {[
            { id: 'hotspot', label: 'Hotspot Mapping' },
            { id: 'predictor', label: 'Risk Predictor' },
            { id: 'drivers', label: 'Driver Analysis' },
            { id: 'policy', label: 'Policy Simulator' },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={cn(
                "py-3 border-b-2 transition-all text-sm font-medium outline-none",
                activeTab === tab.id 
                  ? "border-indigo-600 text-indigo-600 font-bold" 
                  : "border-transparent text-slate-400 hover:text-slate-600"
              )}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Dynamic Content */}
        <div className="flex-1 overflow-y-auto p-8 space-y-8">
          {/* Top Row Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <StatCard 
              title="Active Districts" 
              value={filteredDistricts.length} 
              icon={MapIcon} 
              subValue={selectedState === 'All' ? 'Nationwide view' : `In ${selectedState}`} 
            />
            <StatCard 
              title="Avg current MPI" 
              value={avgMpi.toFixed(3)} 
              icon={BarChart3} 
              delta="2.4% vs prev cycle" 
              deltaType="positive"
            />
            <StatCard 
              title="High Risk Zones" 
              value={highRiskCount} 
              icon={AlertTriangle} 
              subValue={`${((highRiskCount / filteredDistricts.length) * 100).toFixed(1)}% of selection`}
              delta="+1 new district" 
              deltaType="negative"
            />
            <StatCard 
              title="Model Acc" 
              value="92.8%" 
              icon={CheckCircle2} 
              subValue="Confidence score" 
            />
          </div>

          {/* Dynamic Tab Content */}
          <div className="min-h-[600px]">
            <AnimatePresence mode="wait">
              {activeTab === 'hotspot' && (
                <motion.div 
                  key="hotspot"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  className="space-y-6"
                >
                  <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
                    <div className="flex items-center justify-between mb-8">
                      <div>
                        <h3 className="text-lg font-bold text-slate-800 flex items-center gap-2">
                          <MapIcon className="w-5 h-5 text-indigo-600" />
                          Risk Hotspot Mapping
                        </h3>
                        <p className="text-xs text-slate-500 font-medium">Current MPI vs Improvement Velocity</p>
                      </div>
                      <div className="flex items-center gap-4 text-xs font-mono font-medium">
                        <div className="flex items-center gap-1.5"><div className="w-3 h-3 rounded-full bg-emerald-500" /> Low</div>
                        <div className="flex items-center gap-1.5"><div className="w-3 h-3 rounded-full bg-amber-500" /> Med</div>
                        <div className="flex items-center gap-1.5"><div className="w-3 h-3 rounded-full bg-rose-500" /> High</div>
                      </div>
                    </div>
                    
                    <div className="h-[400px] w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                          <XAxis 
                            type="number" 
                            dataKey="mpi_current" 
                            name="Current MPI" 
                            unit="" 
                            domain={[0, 0.4]}
                            label={{ value: 'Current MPI Index', position: 'bottom', offset: -5, fontSize: 10, fontFamily: 'monospace' }}
                            stroke="#94a3b8"
                            fontSize={10}
                          />
                          <YAxis 
                            type="number" 
                            dataKey="mpi_change" 
                            name="MPI Change" 
                            unit="" 
                            domain={[-0.05, 0.15]}
                            label={{ value: 'MPI Improvement Velocity', angle: -90, position: 'left', offset: 10, fontSize: 10, fontFamily: 'monospace' }}
                            stroke="#94a3b8"
                            fontSize={10}
                          />
                          <ZAxis type="number" dataKey="rural_pop_pct" range={[50, 400]} />
                          <Tooltip 
                            cursor={{ strokeDasharray: '3 3' }} 
                            contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)', fontSize: '12px' }}
                          />
                          <Scatter name="Districts" data={filteredDistricts}>
                            {filteredDistricts.map((entry, index) => (
                              <Cell 
                                key={`cell-${index}`} 
                                fill={entry.risk_band === 'High' ? '#f43f5e' : entry.risk_band === 'Medium' ? '#f59e0b' : '#10b981'} 
                                stroke={entry.id === selectedDistrictId ? '#000' : 'transparent'}
                                strokeWidth={2}
                                onClick={() => setSelectedDistrictId(entry.id)}
                                className="cursor-pointer transition-all hover:opacity-80"
                              />
                            ))}
                          </Scatter>
                          {/* Quadrant labels removed to ensure type safety and build success */}

                        </ScatterChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm">
                    <table className="w-full text-left border-collapse">
                      <thead>
                        <tr className="bg-slate-50 border-b border-slate-200">
                          <th className="px-6 py-3 text-[10px] font-mono font-bold text-slate-500 uppercase">District</th>
                          <th className="px-6 py-3 text-[10px] font-mono font-bold text-slate-500 uppercase">State</th>
                          <th className="px-6 py-3 text-[10px] font-mono font-bold text-slate-500 uppercase">Index</th>
                          <th className="px-6 py-3 text-[10px] font-mono font-bold text-slate-500 uppercase">Velocity</th>
                          <th className="px-6 py-3 text-[10px] font-mono font-bold text-slate-500 uppercase">Stagnation Risk</th>
                          <th className="px-1 py-1"></th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100">
                        {filteredDistricts.slice(0, 10).map((d) => (
                          <tr 
                            key={d.id} 
                            onClick={() => setSelectedDistrictId(d.id)}
                            className={cn(
                              "group cursor-pointer hover:bg-blue-50 transition-colors",
                              selectedDistrictId === d.id ? "bg-blue-50/50" : ""
                            )}
                          >
                            <td className="px-6 py-4 text-sm font-bold text-slate-900">{d.district}</td>
                            <td className="px-6 py-4 text-xs font-medium text-slate-500">{d.state}</td>
                            <td className="px-6 py-4 text-sm font-mono tracking-tighter">{d.mpi_current.toFixed(3)}</td>
                            <td className="px-6 py-4 text-xs">
                              <span className={cn(
                                "flex items-center gap-1",
                                d.mpi_change > 0 ? "text-emerald-600" : "text-rose-600"
                              )}>
                                {d.mpi_change > 0 ? <TrendingDown size={14} /> : <TrendingUp size={14} />}
                                {d.mpi_change.toFixed(3)}
                              </span>
                            </td>
                            <td className="px-6 py-4">
                              <div className="flex items-center gap-3">
                                <div className="flex-1 h-1.5 bg-slate-100 rounded-full overflow-hidden max-w-[80px]">
                                  <div 
                                    className={cn(
                                      "h-full rounded-full transition-all duration-500",
                                      d.risk_band === 'High' ? "bg-rose-500" : d.risk_band === 'Medium' ? "bg-amber-500" : "bg-emerald-500"
                                    )}
                                    style={{ width: `${d.stagnation_risk_prob * 100}%` }}
                                  />
                                </div>
                                <span className="text-[10px] font-mono font-bold">{(d.stagnation_risk_prob * 100).toFixed(0)}%</span>
                              </div>
                            </td>
                            <td className="px-1 text-slate-300 group-hover:text-blue-500">
                              <ChevronRight size={16} />
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </motion.div>
              )}

              {activeTab === 'predictor' && currentDistrict && (
                <motion.div 
                  key="predictor"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="grid grid-cols-1 md:grid-cols-3 gap-8"
                >
                  <div className="md:col-span-2 space-y-8">
                    <div className="bg-white p-8 rounded-xl border border-slate-200 shadow-sm relative overflow-hidden">
                      <div className="absolute top-0 right-0 w-32 h-32 bg-indigo-50/30 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2" />
                      <div className="flex items-start justify-between mb-8 relative z-10">
                        <div>
                          <Badge variant={currentDistrict.risk_band === 'High' ? 'error' : currentDistrict.risk_band === 'Medium' ? 'warning' : 'success'}>
                            {currentDistrict.risk_band} RISK PROFILE
                          </Badge>
                          <h2 className="text-4xl font-black tracking-tight mt-6 text-slate-900">
                            {currentDistrict.district.toUpperCase()}
                          </h2>
                          <p className="text-slate-500 text-sm font-bold uppercase tracking-widest mt-1 opacity-60">Regional Tier: {currentDistrict.state}</p>
                        </div>
                        <div className="p-4 bg-slate-50 rounded-2xl border border-slate-200 text-center shadow-inner">
                          <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-1">Base Risk Score</p>
                          <div className="text-4xl font-black text-slate-800">{(currentDistrict.stagnation_risk_prob * 100).toFixed(1)}<span className="text-xl text-slate-400">%</span></div>
                        </div>
                      </div>

                      <div className="grid grid-cols-2 lg:grid-cols-4 gap-8 py-8 border-y border-slate-100">
                        <div className="space-y-1">
                          <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest leading-none">Baseline MPI</p>
                          <p className="text-xl font-bold text-slate-800">{currentDistrict.mpi_previous.toFixed(3)}</p>
                        </div>
                        <div className="space-y-1 text-center">
                          <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest leading-none">Current MPI</p>
                          <p className="text-xl font-bold text-slate-800">{currentDistrict.mpi_current.toFixed(3)}</p>
                        </div>
                        <div className="space-y-1 text-center">
                          <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest leading-none">Net Velocity</p>
                          <p className={cn("text-xl font-bold", currentDistrict.mpi_change > 0 ? "text-emerald-500" : "text-rose-500")}>
                            {(currentDistrict.mpi_change > 0 ? "+" : "")}{currentDistrict.mpi_change.toFixed(3)}
                          </p>
                        </div>
                        <div className="space-y-1 text-right">
                          <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest leading-none">Rurality Index</p>
                          <p className="text-xl font-bold text-slate-800 text-indigo-600">{currentDistrict.rural_pop_pct}%</p>
                        </div>
                      </div>

                      <div className="mt-8 flex items-center gap-4 p-4 bg-indigo-50 border border-indigo-100 rounded-xl">
                        <div className="p-2 bg-indigo-600 rounded-lg shrink-0">
                          <Info className="w-5 h-5 text-white" />
                        </div>
                        <p className="text-xs text-indigo-900 font-medium leading-relaxed">
                          <span className="font-bold">Intervention Note:</span> This district shows {currentDistrict.risk_band.toLowerCase()} risk of stagnation. Current policy inputs suggest expanding infrastructure in Clean Cooking Fuel adoption to break the trend.
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="space-y-6">
                    <div className="bg-slate-900 text-white p-6 rounded-2xl shadow-xl">
                      <h4 className="text-sm font-bold tracking-tight mb-6">Indicator Health Status</h4>
                      <div className="space-y-4">
                        {[
                          { label: 'Sanitation', val: currentDistrict.sanitation_pct },
                          { label: 'Clean Fuel', val: currentDistrict.clean_fuel_pct },
                          { label: 'Schooling', val: currentDistrict.schooling_years_pct },
                          { label: 'Housing', val: currentDistrict.housing_pct },
                          { label: 'Health', val: currentDistrict.health_access_pct },
                        ].map((item) => (
                          <div key={item.label} className="space-y-1.5">
                            <div className="flex justify-between items-center text-[10px] font-mono uppercase tracking-widest">
                              <span>{item.label}</span>
                              <span className={cn(item.val > 70 ? "text-emerald-400" : item.val > 50 ? "text-amber-400" : "text-rose-400")}>
                                {item.val}%
                              </span>
                            </div>
                            <div className="h-1 bg-slate-800 rounded-full overflow-hidden">
                              <motion.div 
                                initial={{ width: 0 }}
                                animate={{ width: `${item.val}%` }}
                                className={cn(
                                  "h-full rounded-full transition-all duration-1000",
                                  item.val > 70 ? "bg-emerald-400" : item.val > 50 ? "bg-amber-400" : "bg-rose-400"
                                )}
                              />
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </motion.div>
              )}

              {activeTab === 'drivers' && (
                <motion.div 
                  key="drivers"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="grid grid-cols-1 md:grid-cols-2 gap-8"
                >
                  <div className="bg-white p-8 border-b-2 border-r-2 border-slate-900 shadow-[8px_8px_0px_0px_rgba(15,23,42,1)]">
                    <h3 className="text-xl font-bold tracking-tight mb-8">Model Sensitivity: Sensitivity Ranking</h3>
                    <div className="h-[400px]">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={FEATURE_IMPORTANCE} layout="vertical" margin={{ left: 20 }}>
                          <XAxis type="number" hide />
                          <YAxis 
                            dataKey="feature" 
                            type="category" 
                            width={100} 
                            fontSize={12} 
                            fontWeight="bold" 
                            stroke="#0f172a"
                            axisLine={false}
                            tickLine={false}
                          />
                          <Tooltip 
                            contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}
                          />
                          <Bar dataKey="importance" fill="#4f46e5" radius={[0, 4, 4, 0]} barSize={24} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                    <p className="mt-8 text-xs text-slate-500 font-medium leading-relaxed italic">
                      * This model derives importance from SHAP values, identifying Clean Fuel and Health Access as primary contributors to progress velocity.
                    </p>
                  </div>

                  <div className="bg-white p-8 border-b-2 border-r-2 border-slate-900 shadow-[8px_8px_0px_0px_rgba(15,23,42,1)]">
                    <h3 className="text-xl font-bold tracking-tight mb-8">Asset Distribution</h3>
                    <div className="space-y-8">
                       <div className="grid grid-cols-2 gap-8">
                          <div className="p-6 bg-slate-50 rounded-2xl border border-dashed border-slate-200">
                             <h5 className="text-[10px] font-mono uppercase tracking-widest text-slate-400 mb-4">Rural Weightage</h5>
                             <div className="text-3xl font-black text-slate-900">{currentDistrict?.rural_pop_pct}%</div>
                             <p className="text-[10px] text-slate-500 mt-2 font-medium italic">High rurality correlates with logistical stagnation risk.</p>
                          </div>
                          <div className="p-6 bg-slate-50 rounded-2xl border border-dashed border-slate-200">
                             <h5 className="text-[10px] font-mono uppercase tracking-widest text-slate-400 mb-4">Financial Inclusion</h5>
                             <div className="text-3xl font-black text-slate-900">{currentDistrict?.bank_access_pct}%</div>
                             <p className="text-[10px] text-slate-500 mt-2 font-medium italic">Base capacity for subsidy absorption.</p>
                          </div>
                       </div>
                       <div className="p-6 bg-blue-50 rounded-2xl border-2 border-blue-600 relative overflow-hidden">
                          <Zap className="absolute -right-4 -bottom-4 w-32 h-32 opacity-5 text-blue-900" />
                          <h5 className="text-xs font-bold text-blue-900 mb-2">Dominant Divergence</h5>
                          <p className="text-xs text-blue-800 leading-relaxed font-medium">
                            The gap between Female Literacy ({currentDistrict?.female_literacy_pct}%) and Health Access ({currentDistrict?.health_access_pct}%) in {currentDistrict?.district} suggests that awareness exceeds service provision.
                          </p>
                       </div>
                    </div>
                  </div>
                </motion.div>
              )}

              {activeTab === 'policy' && scenarioResult && currentDistrict && (
                <motion.div 
                  key="policy"
                  initial={{ opacity: 0, scale: 0.98 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="space-y-8"
                >
                  <div className="grid grid-cols-5 gap-6">
                    {/* Left: Simulation Results */}
                    <div className="col-span-2 bg-white rounded-xl border border-slate-200 shadow-sm p-6 flex flex-col items-center">
                       <h3 className="text-sm font-bold text-slate-800 mb-8 self-start uppercase tracking-wider opacity-60">Stagnation Risk Analysis</h3>
                       
                       <div className="flex-1 flex flex-col items-center justify-center gap-6 w-full py-4">
                          <div className="relative w-48 h-48">
                            <svg viewBox="0 0 100 50" className="w-full">
                              <path d="M 10 50 A 40 40 0 0 1 90 50" fill="none" stroke="#F1F5F9" strokeWidth="10" strokeLinecap="round" />
                              <path 
                                d="M 10 50 A 40 40 0 0 1 90 50" 
                                fill="none" 
                                stroke={scenarioResult.scenarioProb > 0.65 ? "#f43f5e" : scenarioResult.scenarioProb > 0.35 ? "#f59e0b" : "#10b981"} 
                                strokeWidth="10" 
                                strokeLinecap="round" 
                                strokeDasharray={`${(scenarioResult.scenarioProb * 125)}, 251.3`}
                              />
                            </svg>
                            <div className="absolute inset-0 flex flex-col items-center justify-end pb-4">
                              <div className="text-4xl font-bold text-slate-800">{(scenarioResult.scenarioProb * 100).toFixed(1)}%</div>
                              <div className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Scenario Risk</div>
                            </div>
                          </div>
                          <div className="text-center">
                            <p className="text-xs text-slate-500 mb-1">Baseline: <span className="font-bold text-slate-900">{(scenarioResult.baseProb * 100).toFixed(1)}%</span></p>
                            <p className="text-sm text-emerald-600 font-bold italic">{(scenarioResult.baseProb - scenarioResult.scenarioProb) > 0 ? `-${((scenarioResult.baseProb - scenarioResult.scenarioProb) * 100).toFixed(1)}% Reduction Projected` : 'No improvement'}</p>
                          </div>
                       </div>

                       <div className="w-full mt-auto space-y-3">
                          <div className="flex justify-between items-center p-4 rounded-xl bg-indigo-50 border border-indigo-100">
                            <span className="text-xs font-bold text-indigo-900 uppercase tracking-wide">Projected Next cycle MPI</span>
                            <span className="text-base font-black text-indigo-700">{scenarioResult.projectedMpi.toFixed(3)} <span className={cn("text-xs font-bold ml-1", scenarioResult.projectedMpi < currentDistrict.mpi_current ? "text-emerald-600" : "text-slate-400")}>{(scenarioResult.projectedMpi - currentDistrict.mpi_current).toFixed(3)}</span></span>
                          </div>
                       </div>
                    </div>

                    {/* Right: Bar Comparison */}
                    <div className="col-span-3 bg-white rounded-xl border border-slate-200 shadow-sm p-6 flex flex-col">
                      <div className="flex justify-between items-center mb-8">
                        <h3 className="text-sm font-bold text-slate-800 uppercase tracking-wider opacity-60">Effectiveness Benchmark</h3>
                        <div className="flex gap-4">
                          <div className="flex items-center gap-1.5 text-[10px] font-bold text-slate-400 uppercase tracking-tighter transition-all"><span className="w-2.5 h-2.5 rounded-sm bg-slate-200 shadow-sm"></span> Baseline</div>
                          <div className="flex items-center gap-1.5 text-[10px] font-bold text-indigo-600 uppercase tracking-tighter transition-all"><span className="w-2.5 h-2.5 rounded-sm bg-indigo-500 shadow-sm shadow-indigo-100"></span> Scenario</div>
                        </div>
                      </div>

                      <div className="flex-1 flex flex-col justify-between py-2">
                        {[
                          { label: 'Sanitation Expansion', base: currentDistrict.sanitation_pct, scenario: scenarioResult.indicators.sanitation_pct },
                          { label: 'Clean Fuel Adoption', base: currentDistrict.clean_fuel_pct, scenario: scenarioResult.indicators.clean_fuel_pct },
                          { label: 'Schooling Improvement', base: currentDistrict.schooling_years_pct, scenario: scenarioResult.indicators.schooling_years_pct },
                          { label: 'Housing Upgrades', base: currentDistrict.housing_pct, scenario: scenarioResult.indicators.housing_pct },
                          { label: 'Health Access Efficiency', base: currentDistrict.health_access_pct, scenario: scenarioResult.indicators.health_access_pct },
                        ].map(item => (
                          <div key={item.label} className="space-y-2">
                            <div className="flex justify-between text-[11px] font-bold text-slate-500 mb-1">
                              <span>{item.label}</span>
                              <span className="text-indigo-600">+{item.scenario - item.base}%</span>
                            </div>
                            <div className="h-4 w-full flex rounded-lg overflow-hidden border border-slate-100 shadow-inner group">
                              <div className="bg-slate-100/80 h-full transition-all duration-500" style={{ width: `${item.base}%` }}></div>
                              <div className="bg-indigo-500 h-full border-l border-white/20 transition-all duration-700 shadow-[inset_-4px_0_8px_-4px_rgba(255,255,255,0.4)]" style={{ width: `${item.scenario - item.base}%` }}></div>
                            </div>
                          </div>
                        ))}
                      </div>
                      
                      <div className="mt-8 flex justify-end items-center border-t border-slate-100 pt-4">
                         <p className="text-[10px] text-slate-400 font-bold tracking-widest italic opacity-50 uppercase">Interventions Weighted by RF Model Sensitivity</p>
                      </div>
                    </div>
                  </div>

                  <div className="flex justify-center pt-4">
                    <button className="px-10 py-3 bg-indigo-600 text-white rounded-xl font-bold flex items-center gap-2 hover:bg-indigo-700 shadow-xl shadow-indigo-100 transition-all group active:scale-95">
                      Export Intervention Manifest
                      <ArrowRight className="w-4 h-4 translate-x-0 group-hover:translate-x-1 transition-transform" />
                    </button>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>

        <footer className="px-8 py-3 bg-white border-t border-slate-200 flex justify-between items-center text-[10px] font-bold text-slate-400 uppercase tracking-widest shrink-0">
          <div className="flex items-center gap-6">
            <span>PROTOTYPE V1.2 • Unity: MPI ANALYTICS ENGINE</span>
            <div className="h-3 w-px bg-slate-200" />
            <div className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
              Engine: Verified State
            </div>
          </div>
          <div className="flex gap-4">
            <span>REFRESH: 24H</span>
            <span className="text-slate-300">|</span>
            <span>SIU INTEL UNIT</span>
          </div>
        </footer>
      </main>
    </div>
  );
}

