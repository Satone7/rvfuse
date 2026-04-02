int main() {
    int sum = 0;
    for (int i = 0; i < 10; i++) {
        if (i % 2 == 0) {
            sum += i;
        } else {
            sum -= i;
        }
    }
    int j = 0;
    while (j < 5) {
        sum *= 2;
        j++;
    }
    return sum;
}

void _start() {
    main();
    // exit syscall
    register int a0 asm("a0") = 0;
    register int a7 asm("a7") = 93; // __NR_exit
    asm volatile ("ecall" : : "r"(a0), "r"(a7));
}
