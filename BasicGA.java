import java.util.*;
import java.util.function.*;

public class BasicGA {

    /* ======================== Functional interfaces ======================== */
    @FunctionalInterface
    public interface FitnessFunction<G> { double evaluate(List<G> individual); }

    @FunctionalInterface
    public interface CrossoverOp<G> {
        Pair<List<G>, List<G>> apply(List<G> p1, List<G> p2, Random rnd);
    }

    @FunctionalInterface
    public interface MutationOp<G> {
        List<G> apply(List<G> individual, Random rnd, List<G> genePool);
    }

    /* ======================== Small helpers ======================== */
    public static final class Pair<A,B> { public final A a; public final B b; public Pair(A a,B b){this.a=a;this.b=b;} }

    public enum SelectionType { TOURNAMENT, ROULETTE }

    public static final class GAConfig {
        public int populationSize = 50;
        public int generations = 500;
        public double pCrossover = 0.7;
        public double pMutation = 0.01; // per-gene mutation prob for default mutator
        public SelectionType selection = SelectionType.TOURNAMENT;
        public int tournamentK = 3;
        public int elitism = 1; // number of elites copied over each gen
        public int logEvery = 50; // print stats every N generations (<=0 to disable)
        public Long seed = null; // if not null, deterministic
        public Double targetFitness = null; // stop when best >= target
        public Integer stagnationLimit = null; // stop if no improvement for this many generations
    }

    public static final class GAResult<G> {
        public final List<G> best;
        public final double bestFitness;
        public final int generationsRan;
        public final boolean reachedTarget;
        public final long evals;
        public final long runtimeMillis;
        GAResult(List<G> best, double fit, int gens, boolean reachedTarget, long evals, long ms){
            this.best = best; this.bestFitness = fit; this.generationsRan = gens; this.reachedTarget = reachedTarget; this.evals = evals; this.runtimeMillis = ms;
        }
    }

    /* ======================== Core GA ======================== */
    public static <G> GAResult<G> run(
            FitnessFunction<G> fitnessFn,
            List<G> genePool,
            int stateLength,
            GAConfig cfg,
            CrossoverOp<G> crossover,
            MutationOp<G> mutation
    ) {
        Objects.requireNonNull(fitnessFn, "fitnessFn");
        Objects.requireNonNull(genePool, "genePool");
        if (genePool.isEmpty()) throw new IllegalArgumentException("genePool must not be empty");
        if (stateLength <= 0) throw new IllegalArgumentException("stateLength must be > 0");
        Objects.requireNonNull(cfg, "cfg");
        Objects.requireNonNull(crossover, "crossover");
        Objects.requireNonNull(mutation, "mutation");

        Random rnd = cfg.seed == null ? new Random() : new Random(cfg.seed);

        // init population
        List<List<G>> population = new ArrayList<>(cfg.populationSize);
        for (int i=0;i<cfg.populationSize;i++) population.add(randomIndividual(genePool, stateLength, rnd));

        long evals = 0L;
        long t0 = System.currentTimeMillis();

        List<Double> fitnesses = new ArrayList<>(cfg.populationSize);
        for (List<G> ind : population) { double f = fitnessFn.evaluate(ind); fitnesses.add(f); evals++; }
        int bestIdx = argMaxIndex(fitnesses);
        List<G> best = deepCopy(population.get(bestIdx));
        double bestFitness = fitnesses.get(bestIdx);
        double lastImprovementAtFit = bestFitness;
        int lastImprovementGen = 0;

        // main loop
        for (int gen=1; gen<=cfg.generations; gen++) {
            // Elites
            List<List<G>> nextPop = new ArrayList<>(cfg.populationSize);
            if (cfg.elitism > 0) {
                int[] eliteIdx = topKIndices(fitnesses, Math.min(cfg.elitism, population.size()));
                for (int ei : eliteIdx) nextPop.add(deepCopy(population.get(ei)));
            }

            // Offspring
            while (nextPop.size() < cfg.populationSize) {
                List<G> p1 = select(population, fitnesses, cfg, rnd);
                List<G> p2 = select(population, fitnesses, cfg, rnd);
                List<G> c1 = deepCopy(p1), c2 = deepCopy(p2);
                if (rnd.nextDouble() < cfg.pCrossover) {
                    Pair<List<G>,List<G>> ch = crossover.apply(p1, p2, rnd);
                    c1 = ch.a; c2 = ch.b;
                }
                c1 = mutation.apply(c1, rnd, genePool);
                if (nextPop.size() + 1 < cfg.populationSize) {
                    c2 = mutation.apply(c2, rnd, genePool);
                    nextPop.add(c1);
                    nextPop.add(c2);
                } else {
                    nextPop.add(c1);
                }
            }

            population = nextPop;
            fitnesses.clear();
            for (List<G> ind : population) { double f = fitnessFn.evaluate(ind); fitnesses.add(f); evals++; }

            int genBest = argMaxIndex(fitnesses);
            double genBestFit = fitnesses.get(genBest);
            if (genBestFit > bestFitness) {
                bestFitness = genBestFit;
                best = deepCopy(population.get(genBest));
                lastImprovementAtFit = bestFitness;
                lastImprovementGen = gen;
            }

            // logging
            if (cfg.logEvery > 0 && (gen % cfg.logEvery == 0 || gen == cfg.generations)) {
                System.out.printf(Locale.ROOT, "Gen %d | best=%.4f | mean=%.4f | max-ind=%d%n", gen, bestFitness, mean(fitnesses), genBest);
            }

            // early stopping: target fitness
            if (cfg.targetFitness != null && bestFitness >= cfg.targetFitness) {
                long ms = System.currentTimeMillis() - t0;
                return new GAResult<>(best, bestFitness, gen, true, evals, ms);
            }
            // early stopping: stagnation
            if (cfg.stagnationLimit != null && (gen - lastImprovementGen) >= cfg.stagnationLimit) {
                long ms = System.currentTimeMillis() - t0;
                return new GAResult<>(best, bestFitness, gen, false, evals, ms);
            }
        }

        long ms = System.currentTimeMillis() - t0;
        return new GAResult<>(best, bestFitness, cfg.generations, false, evals, ms);
    }

    /* ======================== Selections ======================== */
    private static <G> List<G> select(List<List<G>> pop, List<Double> fits, GAConfig cfg, Random rnd) {
        switch (cfg.selection) {
            case TOURNAMENT: return tournamentSelect(pop, fits, cfg.tournamentK, rnd);
            case ROULETTE: return rouletteSelect(pop, fits, rnd);
            default: throw new IllegalArgumentException("Unknown selection type");
        }
    }

    private static <G> List<G> tournamentSelect(List<List<G>> pop, List<Double> fits, int k, Random rnd) {
        int n = pop.size();
        int bestIdx = rnd.nextInt(n);
        double bestFit = fits.get(bestIdx);
        for (int i=1;i<k;i++) {
            int idx = rnd.nextInt(n);
            if (fits.get(idx) > bestFit) { bestIdx = idx; bestFit = fits.get(idx); }
        }
        return pop.get(bestIdx);
    }

    private static <G> List<G> rouletteSelect(List<List<G>> pop, List<Double> fits, Random rnd) {
        double min = fits.stream().min(Double::compare).orElse(0.0);
        double shift = min < 0 ? -min : 0.0; // ensure non-negative
        double sum = 0.0; for (double f : fits) sum += (f + shift);
        if (sum <= 0) return pop.get(rnd.nextInt(pop.size()));
        double r = rnd.nextDouble() * sum;
        double acc = 0.0;
        for (int i=0;i<pop.size();i++) {
            acc += fits.get(i) + shift;
            if (acc >= r) return pop.get(i);
        }
        return pop.get(pop.size()-1);
    }

    /* ======================== Default operators ======================== */
    public static <G> CrossoverOp<G> onePointCrossover() {
        return (p1, p2, rnd) -> {
            int n = p1.size();
            if (n <= 1) return new Pair<>(new ArrayList<>(p1), new ArrayList<>(p2));
            int point = 1 + rnd.nextInt(n - 1);
            List<G> c1 = new ArrayList<>(n);
            List<G> c2 = new ArrayList<>(n);
            c1.addAll(p1.subList(0, point));
            c1.addAll(p2.subList(point, n));
            c2.addAll(p2.subList(0, point));
            c2.addAll(p1.subList(point, n));
            return new Pair<>(c1, c2);
        };
    }

    /** Random-reset mutation: with per-gene probability p, replace by random gene from pool. */
    public static <G> MutationOp<G> randomResetMutation(double perGeneProb) {
        if (perGeneProb < 0 || perGeneProb > 1) throw new IllegalArgumentException("perGeneProb must be in [0,1]");
        return (ind, rnd, pool) -> {
            List<G> child = new ArrayList<>(ind);
            for (int i=0;i<child.size();i++) if (rnd.nextDouble() < perGeneProb) child.set(i, pool.get(rnd.nextInt(pool.size())));
            return child;
        };
    }

    /* ======================== Utilities ======================== */
    private static <G> List<G> randomIndividual(List<G> genePool, int len, Random rnd) {
        List<G> ind = new ArrayList<>(len);
        for (int i=0;i<len;i++) ind.add(genePool.get(rnd.nextInt(genePool.size())));
        return ind;
    }

    private static int argMaxIndex(List<Double> vals) {
        int best = 0; double bestVal = vals.get(0);
        for (int i=1;i<vals.size();i++) if (vals.get(i) > bestVal) { best = i; bestVal = vals.get(i); }
        return best;
    }

    private static double mean(List<Double> vals) {
        double s = 0.0; for (double v : vals) s += v; return s / vals.size();
    }

    private static int[] topKIndices(List<Double> vals, int k) {
        int n = vals.size();
        Integer[] idx = new Integer[n];
        for (int i=0;i<n;i++) idx[i] = i;
        Arrays.sort(idx, (i,j) -> Double.compare(vals.get(i), vals.get(j))); // ascending
        int[] out = new int[Math.min(k, n)];
        for (int t=0; t<out.length; t++) out[t] = idx[n-1-t];
        return out;
    }

    private static <T> List<T> deepCopy(List<T> list) {
        return new ArrayList<>(list); // shallow elements, but list copy is enough for immutable genes
    }

    /* ======================== Demo: MaxOnes ======================== */
    public static void main(String[] args) {
        // Maximize number of ones in a 0/1 string of length 64
        List<Integer> genePool = Arrays.asList(0,1);
        int length = 64;

        FitnessFunction<Integer> fitness = ind -> {
            int sum = 0; for (int v : ind) sum += v; return sum; // count ones
        };

        GAConfig cfg = new GAConfig();
        cfg.populationSize = 80;
        cfg.generations = 1000;
        cfg.pCrossover = 0.9;
        cfg.pMutation = 0.02; // used only for logging; actual per-gene prob set in mutation op below
        cfg.selection = SelectionType.TOURNAMENT;
        cfg.tournamentK = 3;
        cfg.elitism = 2;
        cfg.logEvery = 50;
        cfg.seed = 42L; // deterministic runs
        cfg.targetFitness = (double) length; // stop at perfect solution
        cfg.stagnationLimit = 200; // optional early stop

        CrossoverOp<Integer> xover = onePointCrossover();
        MutationOp<Integer> mut = randomResetMutation(0.01); // per-gene mutation probability

        GAResult<Integer> res = run(fitness, genePool, length, cfg, xover, mut);

        System.out.println("==== RESULT ====");
        System.out.println("Best fitness: " + res.bestFitness + "/" + length);
        System.out.println("Generations: " + res.generationsRan + ", evals: " + res.evals + ", time: " + res.runtimeMillis + " ms");
        System.out.println("Reached target: " + res.reachedTarget);
        System.out.println("Best individual: " + res.best);
    }
}
