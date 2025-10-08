
def greeting_templates(subject, grade):
    templates = [
    f"""Hello! I'm your AI Teaching Assistant. I'm excited to help you prepare for teaching {subject} to {grade} students!

                Together, we'll explore:
                • Your deep understanding of {subject}
                • Effective pedagogical strategies for {grade} level
                • Culturally relevant teaching approaches for Nigerian students
                • Real-world scenarios with students, parents, and administrators

                By the end, you'll receive a comprehensive readiness assessment with personalized feedback.

                Let's begin! To start, tell me: What specific subject or concept within {subject} will you be focusing on in your lesson?""",

                f"""Welcome! I'm here as your personal teaching coach for your upcoming {subject} lesson with {grade} students.

                We'll work through:
                1. Deep exploration of your {subject} content knowledge
                2. Engaging teaching strategies for {grade} learners
                3. Cultural relevance and local context integration
                4. Practice scenarios to build your confidence

                I'll provide detailed feedback to ensure you're fully prepared.

                To kick things off: What's the main concept or subject you'll be teaching within {subject}?""",

                f"""Hi there! Ready to prepare for your {subject} lesson with {grade} students? I'm your AI Teaching Assistant, and I'll guide you through a comprehensive preparation process.

                Here's what we'll cover:
                → Content mastery in {subject}
                → Student engagement strategies for {grade}
                → Culturally responsive teaching methods
                → Roleplay practice with realistic scenarios

                At the end, you'll get a detailed assessment of your readiness.

                Let's start: What specific aspect or subject of {subject} will your lesson focus on?"""
                ,
    
    f"""Hello! Excited to help you get ready for teaching {subject} to your {grade} students.  

    Together, we'll dive into:
    • Mastery of {subject} concepts
    • Effective and fun teaching strategies for {grade} learners
    • Contextual examples relevant to Nigerian classrooms
    • Interactive scenarios to practice your teaching  

    At the end, you'll receive tailored feedback to sharpen your lesson delivery.  

    Let's get started! Which part of {subject} do you want to focus on first?""",

    f"""Greetings! I'm here to assist you in preparing an outstanding {subject} lesson for {grade} students.  

    Our session will cover:
    1. Deep understanding of key {subject} subjects
    2. Engaging techniques for your {grade} learners
    3. Incorporating cultural and local relevance
    4. Scenario-based practice to boost confidence  

    Before we dive in, tell me: What main concept in {subject} are we focusing on today?""",

    f"""Hi! Ready to make your {subject} lesson with {grade} students a success? I'm your AI Teaching Assistant here to guide you.  

    We'll focus on:
    → Strong grasp of {subject}
    → Classroom engagement strategies
    → Culturally responsive teaching methods
    → Practical roleplay and feedback  

    To begin, which specific subject or concept within {subject} should we start with?""",

    f"""Welcome back! Let's prepare a fantastic {subject} lesson for your {grade} students.  

    Today, we'll work on:
    • Understanding the core of {subject}
    • Active teaching strategies for {grade}
    • Real-life examples suited to your students' context
    • Simulated teaching exercises with feedback  

    What's the first concept in {subject} that you'd like to focus on?""",

    f"""Hello there! Excited to support you in teaching {subject} to {grade} learners.  

    In our session, you'll get guidance on:
    1. Content mastery of {subject}
    2. Creative ways to engage your {grade} students
    3. Making lessons culturally meaningful
    4. Practical readiness through interactive exercises  

    Let's start: Which subject or concept within {subject} are we tackling first?""",
    
    f"""Hello! I'm thrilled to help you prepare for teaching {subject} to {grade} students.  

    Together, we'll:
    • Dive deep into {subject} concepts
    • Explore strategies to make learning engaging
    • Incorporate culturally relevant examples
    • Practice scenarios to boost your teaching confidence  

    To begin, which subject in {subject} shall we focus on first?""",

    f"""Hi there! Ready to make your {subject} lesson for {grade} students unforgettable?  

    We'll cover:
    → Content mastery in {subject}
    → Innovative teaching techniques
    → Realistic classroom roleplays
    → Feedback to refine your approach  

    What's the first concept you'd like to tackle today?""",

    f"""Greetings! Your {subject} lesson preparation for {grade} learners starts here.  

    In this session, you'll get guidance on:
    1. Understanding key concepts
    2. Engaging students effectively
    3. Integrating local and cultural context
    4. Practicing real-life teaching scenarios  

    To start, which area of {subject} are you focusing on?""",

    f"""Hey! Excited to guide you through your {subject} lesson for {grade} students.  

    We'll work on:
    • Mastery of {subject} subjects
    • Making lessons interactive and fun
    • Tailoring teaching to your students' context
    • Roleplay exercises to boost readiness  

    Which part of {subject} should we dive into first?""",

    f"""Welcome! Let’s prepare a comprehensive {subject} lesson for {grade} students.  

    Today, we’ll explore:
    → In-depth understanding of {subject}
    → Effective classroom strategies
    → Culturally responsive teaching
    → Practice teaching scenarios  

    What subject in {subject} will be the focus of your lesson?""",

    f"""Hello! I'm your AI Teaching Assistant, ready to support your {subject} lesson planning for {grade} students.  

    Together, we’ll:
    • Strengthen your content knowledge
    • Learn methods to engage learners
    • Make lessons culturally relevant
    • Simulate real classroom interactions  

    Let’s begin! Which concept should we start with?""",

    f"""Hi! Excited to help you get classroom-ready for your {subject} lesson with {grade} students.  

    Our session includes:
    1. Deep dive into {subject}
    2. Engagement and teaching strategies
    3. Culturally aligned teaching examples
    4. Practice exercises and feedback  

    What’s the first {subject} concept to focus on?""",

    f"""Greetings! Let’s make your {subject} lesson for {grade} students engaging and effective.  

    We’ll cover:
    • Thorough understanding of {subject}
    • Classroom interaction techniques
    • Culturally meaningful examples
    • Simulated teaching scenarios  

    Which subject should we tackle first?""",

    f"""Hi there! Ready to plan a winning {subject} lesson for {grade} students?  

    We’ll work on:
    → Core {subject} concepts
    → Creative teaching methods
    → Contextualized examples for Nigerian classrooms
    → Real-life practice exercises  

    Which part of {subject} do you want to start with?""",

    f"""Hello! Let’s kick off your preparation for a {subject} lesson with {grade} students.  

    Together, we’ll explore:
    • Content expertise in {subject}
    • Strategies to captivate your students
    • Integrating cultural relevance
    • Practice teaching scenarios with feedback  

    What subject in {subject} would you like to focus on first?""",

    f"""Hi! Excited to help you sharpen your {subject} lesson for {grade} learners.  

    In this session, you’ll get:
    1. Deep knowledge of {subject}
    2. Student engagement strategies
    3. Cultural and local context integration
    4. Practice teaching with feedback  

    Which concept shall we begin with?""",

    f"""Hey there! Let’s make your {subject} lesson for {grade} students dynamic and effective.  

    Today, we’ll focus on:
    • Understanding {subject} deeply
    • Making learning interactive
    • Contextually relevant examples
    • Scenario-based practice  

    Where shall we start in {subject}?""",

    f"""Greetings! I’m here to guide you through your {subject} lesson planning for {grade} students.  

    We’ll cover:
    → Content mastery
    → Effective teaching techniques
    → Cultural relevance
    → Roleplay practice  

    Which subject in {subject} are we focusing on first?""",

    f"""Welcome! Let’s prepare a powerful {subject} lesson for {grade} learners.  

    You’ll learn:
    • In-depth {subject} knowledge
    • Engaging strategies for your classroom
    • Incorporating local context
    • Hands-on practice with scenarios  

    To start, which part of {subject} should we tackle first?""",

    f"""Hi! Ready to plan a great {subject} lesson for {grade} students?  

    In this session, we’ll focus on:
    1. Mastering key concepts
    2. Boosting classroom engagement
    3. Using culturally relevant examples
    4. Practice exercises with feedback  

    Which concept in {subject} do you want to start with?"""
    ]

    

    return templates

