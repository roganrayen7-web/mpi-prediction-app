
export interface DistrictIndicators {
  sanitation_pct: number;
  clean_fuel_pct: number;
  schooling_years_pct: number;
  housing_pct: number;
  bank_access_pct: number;
  female_literacy_pct: number;
  rural_pop_pct: number;
  health_access_pct: number;
}

export interface District extends DistrictIndicators {
  id: string;
  district: string;
  state: string;
  mpi_current: number;
  mpi_previous: number;
  mpi_change: number;
}

export const STATES = [
  "Tamil Nadu", "Bihar", "Odisha", "Assam"
];

const DISTRICTS_MAPPING: Record<string, string[]> = {
  "Tamil Nadu": ["Madurai", "Virudhunagar", "Theni", "Dindigul", "Sivaganga", "Ramanathapuram", "Salem", "Dharmapuri", "Krishnagiri", "Cuddalore"],
  "Bihar": ["Gaya", "Purnia", "Kishanganj", "Araria", "Katihar"],
  "Odisha": ["Koraput", "Kalahandi", "Malkangiri", "Nabarangpur", "Rayagada"],
  "Assam": ["Dhubri", "Barpeta", "Goalpara", "Nagaon", "Karimganj"]
};

export const generateMockData = (): District[] => {
  const data: District[] = [];
  
  Object.entries(DISTRICTS_MAPPING).forEach(([state, districts]) => {
    districts.forEach(districtName => {
      const mpi_previous = parseFloat((Math.random() * (0.35 - 0.05) + 0.05).toFixed(3));
      const mpi_current = parseFloat((mpi_previous * (Math.random() * (1.1 - 0.7) + 0.7)).toFixed(3));
      
      data.push({
        id: `${state}-${districtName}`,
        district: districtName,
        state: state,
        mpi_current,
        mpi_previous,
        mpi_change: parseFloat((mpi_previous - mpi_current).toFixed(3)),
        sanitation_pct: Math.floor(Math.random() * 58 + 40),
        clean_fuel_pct: Math.floor(Math.random() * 70 + 25),
        schooling_years_pct: Math.floor(Math.random() * 61 + 35),
        housing_pct: Math.floor(Math.random() * 65 + 30),
        bank_access_pct: Math.floor(Math.random() * 54 + 45),
        female_literacy_pct: Math.floor(Math.random() * 55 + 40),
        rural_pop_pct: Math.floor(Math.random() * 80 + 15),
        health_access_pct: Math.floor(Math.random() * 60 + 35),
      });
    });
  });
  
  return data;
};

// Simplified Random Forest mock: weighted probability
export const predictStagnationRisk = (district: DistrictIndicators, mpiPrevious: number): number => {
  // Logic: 
  // Lower sanitation, clean fuel, health access -> Higher risk
  // High previous MPI -> usually higher chance of improvement, but here we model stagnation
  let score = 0;
  score += (100 - district.sanitation_pct) * 0.15;
  score += (100 - district.clean_fuel_pct) * 0.18;
  score += (100 - district.health_access_pct) * 0.18;
  score += (100 - district.schooling_years_pct) * 0.12;
  score += district.rural_pop_pct * 0.1;
  score += mpiPrevious * 100 * 0.5;
  
  // Normalize to 0-1
  const prob = Math.min(0.98, Math.max(0.02, score / 120));
  return prob;
};

export const getRiskBand = (prob: number): "Low" | "Medium" | "High" => {
  if (prob < 0.35) return "Low";
  if (prob < 0.65) return "Medium";
  return "High";
};

export const FEATURE_IMPORTANCE = [
  { feature: "Clean Fuel", importance: 0.22 },
  { feature: "Health Access", importance: 0.19 },
  { feature: "Sanitation", importance: 0.17 },
  { feature: "Schooling", importance: 0.14 },
  { feature: "Housing", importance: 0.11 },
  { feature: "Rural Pop", importance: 0.09 },
  { feature: "Bank Access", importance: 0.08 }
].sort((a, b) => b.importance - a.importance);
