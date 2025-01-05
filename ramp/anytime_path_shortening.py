import threading
from concurrent.futures import ThreadPoolExecutor
import ompl.base as ob
import ompl.geometric as og
import threading
import multiprocessing


class AnytimePathShortening(ob.Planner):
    def __init__(self, si):
        super(AnytimePathShortening, self).__init__(si, "AnytimePathShortening")

        # Planner settings
        # self.specs_.approximateSolutions = True
        # self.specs_.multithreaded = True
        # self.specs_.optimizingPaths = True

        # Instance variables
        self.planners_ = []
        self.shortcut_ = True
        self.hybridize_ = True
        self.max_hybrid_paths_ = 24
        self.default_num_planners_ = max(1, multiprocessing.cpu_count())
        self.best_cost_ = ob.Cost(float("inf"))
        self.lock_ = threading.Lock()
        self.invalid_start_state_count_ = 0
        self.invalid_goal_count_ = 0
        self.invalid_start_goal_lock_ = threading.Lock()
        self.si_ = si

    def solve(self, ptc):
        self.pdef_ = self.getProblemDefinition()
        threads = []
        phybrid = og.PathHybridization(self.getSpaceInformation())

        # Get optimization objective
        opt = self.pdef_.getOptimizationObjective()
        if not opt:
            opt = ob.PathLengthOptimizationObjective(self.si_)
            self.pdef_.setOptimizationObjective(opt)
        print(self.pdef_)

        # Clear any previous planning data
        self.clear()

        # Start planner threads
        for planner in self.planners_:
            thread = threading.Thread(target=self.thread_solve, args=(planner, ptc))
            thread.start()
            threads.append(thread)

        ps = og.PathSimplifier(self.si_)
        sln = None
        prev_last_path = None
        prev_sol_count = 0

        while not ptc():
            # Check if we found a good enough solution
            if opt.isSatisfied(self.best_cost_):
                print("Best cost satisfied")
                ptc.terminate()
                break

            # Hybridize paths if enabled
            sol_count = self.pdef_.getSolutionCount()
            if self.hybridize_ and not ptc() and sol_count > 1:
                paths = self.pdef_.getSolutions()
                num_paths = min(sol_count, self.max_hybrid_paths_)
                last_path = paths[num_paths - 1].path_.get()

                if last_path != prev_last_path or (
                    prev_sol_count < sol_count and sol_count <= self.max_hybrid_paths_
                ):
                    for j in range(num_paths):
                        if ptc():
                            break
                        phybrid.recordPath(paths[j].path_, False)

                    phybrid.computeHybridPath()
                    sln = phybrid.getHybridPath()
                    prev_last_path = last_path
                else:
                    sln = self.pdef_.getSolutionPath()
                prev_sol_count = sol_count
            elif sol_count > 0:
                sln = self.pdef_.getSolutionPath()

            if sln:
                path_copy = og.PathGeometric(sln)
                if self.shortcut_:
                    if not ps.simplify(path_copy, ptc, True):
                        path_copy = og.PathGeometric(sln)
                self.add_path(path_copy, self)

            if self.hybridize_ and phybrid.pathCount() >= self.max_hybrid_paths_:
                phybrid.clear()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Determine final status
        status = ob.PlannerStatus.UNKNOWN
        if self.invalid_start_state_count_ > 0:
            status = ob.PlannerStatus.INVALID_START
        elif self.invalid_goal_count_ > 0:
            status = ob.PlannerStatus.INVALID_GOAL

        if self.pdef_.getSolutionCount() > 0:
            return ob.PlannerStatus(True, False)
        return ob.PlannerStatus(status)

    def clear(self):
        super(AnytimePathShortening, self).clear()
        for planner in self.planners_:
            planner.clear()
        self.best_cost_ = ob.Cost(float("inf"))
        self.invalid_start_state_count_ = 0
        self.invalid_goal_count_ = 0

    def add_path(self, path, planner):
        opt = self.pdef_.getOptimizationObjective()
        path_cost = path.cost(opt)

        with self.lock_:
            if opt.isCostBetterThan(path_cost, self.best_cost_):
                print(path_cost.value(), self.best_cost_.value())
                self.best_cost_ = path_cost
                self.pdef_.addSolutionPath(path, False, 0.0, planner.getName())
            elif planner != self:
                self.pdef_.addSolutionPath(path, False, 0.0, planner.getName())

    def thread_solve(self, planner, ptc):
        # Create local clone of problem definition
        pdef = self.pdef_.clone()
        ps = og.PathSimplifier(self.si_)

        planner.setProblemDefinition(pdef)
        while not ptc():
            status = planner.solve(ptc)
            if status.getStatus() == ob.PlannerStatus.EXACT_SOLUTION:
                sln = pdef.getSolutionPath()
                path_copy = og.PathGeometric(sln)
                if self.shortcut_:
                    ps.partialShortcutPath(path_copy)
                self.add_path(path_copy, planner)
            elif (
                status == ob.PlannerStatus.INVALID_START
                or status == ob.PlannerStatus.INVALID_GOAL
                or status == ob.PlannerStatus.UNRECOGNIZED_GOAL_TYPE
            ):
                with self.invalid_start_goal_lock_:
                    if status == ob.PlannerStatus.INVALID_START:
                        self.invalid_start_state_count_ += 1
                    elif status == ob.PlannerStatus.INVALID_GOAL:
                        self.invalid_goal_count_ += 1
                planner.clear()
                pdef.clearSolutionPaths()
                break

            planner.clear()
            pdef.clearSolutionPaths()

    def setup(self):
        super(AnytimePathShortening, self).setup()

        if not self.planners_:
            self.planners_ = []
            for _ in range(self.default_num_planners_):
                planner = og.RRTConnect(self.si_)  # Default planner
                planner.setProblemDefinition(self.pdef_)
                self.planners_.append(planner)

        for planner in self.planners_:
            planner.setup()

    # Getter/setter methods
    def is_shortcutting(self):
        return self.shortcut_

    def set_shortcut(self, shortcut):
        self.shortcut_ = shortcut

    def is_hybridizing(self):
        return self.hybridize_

    def set_hybridize(self, hybridize):
        self.hybridize_ = hybridize

    def max_hybridization_paths(self):
        return self.max_hybrid_paths_

    def set_max_hybridization_path(self, max_path_count):
        self.max_hybrid_paths_ = max_path_count

    def get_num_planners(self):
        return len(self.planners_)

    def get_planner(self, idx):
        return self.planners_[idx]

    def add_planner(self, planner):
        if planner and planner.getSpaceInformation() != self.si_:
            return

        # Ensure planner is unique
        # for p in self.planners_:
        #     if planner == p:
        #         return

        self.planners_.append(planner)
