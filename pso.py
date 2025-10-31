import numpy as np
import matplotlib.pyplot as plt

# Problem Definition
# - D: số chiều (dimensions)
# - LB, UB: cận dưới và cận trên của không gian tìm kiếm
class Problem:
    def __init__(self, D, LB, UB):
        self.D = D      
        self.LB = LB    
        self.UB = UB    

# Giải mã (decode): chuyển vị trí chuẩn hóa [0,1] về khoảng [LB, UB]
def decode(position, problem: Problem):
    x = position * (problem.UB - problem.LB) + problem.LB
    return np.clip(x, problem.LB, problem.UB)

# Hàm đánh giá (fitness function): sử dụng hàm Rosenbrock
def get_fitness(x):
    # Hàm Rosenbrock chuẩn:
    # f(x) = Σ[100*(x[i+1] - x[i]^2)^2 + (1 - x[i])^2]
    fx = np.sum(100 * (x[1:] - np.square(x[:-1]))**2 + (1 - x[:-1])**2)
    return fx

# Mỗi hạt trong PSO có:
# - position: vị trí hiện tại
# - velocity: vận tốc
# - fitness: giá trị hàm mục tiêu
# - bestPosition, bestFitness: vị trí và giá trị tốt nhất từng đạt được (local best)
class Individual:
    def __init__(self):
        self.position = None
        self.velocity = None
        self.fitness = None
        self.bestFitness = None
        self.bestPosition = None

    # Khởi tạo ngẫu nhiên một cá thể
    def gen_indi(self, problem: Problem):
        self.velocity = np.random.uniform(-0.1, 0.1, problem.D)
        self.position = np.random.uniform(0.0, 1.0, problem.D)
        x = decode(self.position, problem)
        self.fitness = get_fitness(x)
        self.bestFitness = self.fitness
        self.bestPosition = self.position.copy()

    # Tính lại fitness và cập nhật best cá nhân
    def cal_fitness(self, problem: Problem):
        x = decode(self.position, problem)
        self.fitness = get_fitness(x)
        if self.fitness < self.bestFitness:
            self.bestFitness = self.fitness
            self.bestPosition = self.position.copy()

    # Cập nhật vận tốc và vị trí theo công thức PSO
    def move(self, global_best_position, w, c1, c2):
        # r1, r2 là các số ngẫu nhiên trong [0,1]
        r1 = np.random.rand()
        r2 = np.random.rand()
        self.velocity = (w * self.velocity
                         + c1 * r1 * (self.bestPosition - self.position)
                         + c2 * r2 * (global_best_position - self.position))
        # Cập nhật vị trí
        self.position += self.velocity
        self.position = np.clip(self.position, 0, 1)  # Giữ trong [0,1]

    def __repr__(self):
        return (f"position: {self.position}\n"
                f"fitness: {self.fitness}\n"
                f"bestPosition: {self.bestPosition}\n"
                f"bestFitness: {self.bestFitness}\n")

# Swarm (Population)
# Quần thể chứa nhiều cá thể (hạt)
# Mỗi thế hệ, PSO sẽ tìm global_best (toàn cục)
class Population:
    def __init__(self, pop_size, problem: Problem):
        self.pop_size = pop_size
        self.problem = problem
        self.list_indi = []
        self.global_best_position = None
        self.global_best_fitness = np.inf

    # Khởi tạo toàn bộ quần thể
    def gen_pop(self):
        for i in range(self.pop_size):
            indi = Individual()
            indi.gen_indi(self.problem)
            self.list_indi.append(indi)
            # Cập nhật global best nếu cần
            if indi.fitness < self.global_best_fitness:
                self.global_best_fitness = indi.fitness
                self.global_best_position = indi.position.copy()

# PSO Algorithm
# Thực hiện tối ưu hóa qua nhiều thế hệ
def PSO(problem: Problem, pop_size, max_gen, w, c1, c2):
    pop = Population(pop_size, problem)
    pop.gen_pop()          # Bước 1: khởi tạo quần thể
    history = []           # Lưu lịch sử fitness để vẽ đồ thị hội tụ

    for g in range(max_gen):
        # Bước 2: duyệt qua từng cá thể
        for indi in pop.list_indi:
            indi.move(pop.global_best_position, w, c1, c2)   # cập nhật vị trí
            indi.cal_fitness(problem)                        # tính fitness mới

            # Bước 3: cập nhật best toàn cục
            if indi.fitness < pop.global_best_fitness:
                pop.global_best_fitness = indi.fitness
                pop.global_best_position = indi.position.copy()

        # Ghi lại giá trị tốt nhất của thế hệ này
        history.append(pop.global_best_fitness)

    return history, pop.global_best_position

# Run PSO
D = 10      # số chiều (10 biến)
LB = -50    # giới hạn dưới
UB = 50     # giới hạn trên
problem = Problem(D, LB, UB)

pop_size = 20
max_gen = 300
w = 0.7     # hệ số quán tính
c1 = 1.4    # hệ số nhận thức (local)
c2 = 1.4    # hệ số xã hội (global)

# Chạy thuật toán PSO
fitness_history, solution = PSO(problem, pop_size, max_gen, w, c1, c2)

# Visualization
# In ra giá trị fitness tốt nhất mỗi thế hệ
for i, fit in enumerate(fitness_history):
    print(f"Generation {i:03d}, Best Fitness = {fit:.6f}")

# Vẽ đồ thị hội tụ
plt.plot(fitness_history, color='blue')
plt.title("PSO Convergence on Rosenbrock Function")
plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.grid(True)
plt.show()

print("\nBest solution position (normalized in [0,1]):")
print(solution)
