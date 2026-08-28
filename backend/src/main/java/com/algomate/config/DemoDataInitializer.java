package com.algomate.config;

import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.core.annotation.Order;
import org.springframework.stereotype.Component;

import com.algomate.domain.AppUser;
import com.algomate.repository.AppUserRepository;

@Component
@Order(10)
public class DemoDataInitializer implements ApplicationRunner {
    private final AppUserRepository userRepository;

    public DemoDataInitializer(AppUserRepository userRepository) {
        this.userRepository = userRepository;
    }

    @Override
    public void run(ApplicationArguments args) {
        userRepository.findByUsername("demo")
                .orElseGet(() -> userRepository.save(new AppUser("demo", "算法学习者")));
    }
}

