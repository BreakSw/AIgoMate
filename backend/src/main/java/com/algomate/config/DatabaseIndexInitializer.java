package com.algomate.config;

import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.core.annotation.Order;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Component;

@Component
@Order(5)
public class DatabaseIndexInitializer implements ApplicationRunner {
    private final JdbcTemplate jdbcTemplate;

    public DatabaseIndexInitializer(JdbcTemplate jdbcTemplate) {
        this.jdbcTemplate = jdbcTemplate;
    }

    @Override
    public void run(ApplicationArguments args) {
        jdbcTemplate.execute("""
                CREATE INDEX IF NOT EXISTS idx_chat_session_user_updated
                ON chat_session(user_id, updated_at DESC)
                """);
        jdbcTemplate.execute("""
                CREATE INDEX IF NOT EXISTS idx_chat_message_session_created
                ON chat_message(session_id, created_at ASC)
                """);
        jdbcTemplate.execute("""
                CREATE INDEX IF NOT EXISTS idx_intent_analysis_session_created
                ON intent_analysis(session_id, created_at ASC)
                """);
        jdbcTemplate.execute("""
                CREATE INDEX IF NOT EXISTS idx_intent_analysis_primary_intent
                ON intent_analysis(primary_intent)
                """);
        jdbcTemplate.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_intent_analysis_assistant_message
                ON intent_analysis(assistant_message_id)
                """);
        jdbcTemplate.execute("PRAGMA optimize");
    }
}
